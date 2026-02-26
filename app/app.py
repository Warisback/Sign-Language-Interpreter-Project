import socket
import threading
import time
import queue
from collections import deque
from datetime import datetime
from pathlib import Path

import speech_recognition as sr
from flask import Flask, jsonify, render_template


app = Flask(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT_PATH = REPO_ROOT / "speech_to_text" / "transcript.txt"
MIC_DEVICE_INDEX = None  # Set to an integer to force a specific microphone.


state_lock = threading.Lock()
latest_error = ""
captions = deque(maxlen=3)
recording = False
stop_event = threading.Event()
listener_thread = None
transcriber_thread = None
audio_q = queue.Queue(maxsize=32)


def now_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def append_transcript(line: str) -> None:
    try:
        TRANSCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRANSCRIPT_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def add_caption(text: str) -> None:
    with state_lock:
        captions.append(text)


def set_error(error: str = "") -> None:
    global latest_error
    with state_lock:
        latest_error = error


def mark_recording(value: bool) -> None:
    global recording
    with state_lock:
        recording = value


def is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionResetError, TimeoutError, socket.timeout, socket.gaierror)):
        return True
    message = str(exc).lower()
    return "10054" in message or "connection reset" in message or "timed out" in message


def should_run() -> bool:
    with state_lock:
        return recording and not stop_event.is_set()


def listener_worker() -> None:
    global recording, listener_thread
    r = sr.Recognizer()
    r.dynamic_energy_threshold = True
    r.pause_threshold = 0.5
    r.non_speaking_duration = 0.3

    try:
        microphone = sr.Microphone(device_index=MIC_DEVICE_INDEX)
        with microphone as source:
            r.adjust_for_ambient_noise(source, duration=1)
    except Exception as exc:
        set_error(error=f"Microphone init failed: {exc}")
        mark_recording(False)
        stop_event.set()
        with state_lock:
            listener_thread = None
        return

    with microphone as source:
        while should_run():
            try:
                audio = r.listen(source, timeout=2, phrase_time_limit=3)
                try:
                    audio_q.put(audio, timeout=0.5)
                except queue.Full:
                    continue
            except sr.WaitTimeoutError:
                continue
            except Exception as exc:
                set_error(error=f"Listener error: {type(exc).__name__}: {exc}")
                time.sleep(0.2)

    with state_lock:
        listener_thread = None


def transcriber_worker() -> None:
    global transcriber_thread
    r = sr.Recognizer()

    while should_run() or not audio_q.empty():
        try:
            audio = audio_q.get(timeout=0.5)
            text = r.recognize_google(audio, language="en-GB")
            line = f"[{now_ts()}] {text}"
            append_transcript(line)
            add_caption(line)
            set_error(error="")
        except queue.Empty:
            continue
        except sr.UnknownValueError:
            continue
        except sr.RequestError as exc:
            add_caption(f"[{now_ts()}] Speech API error: {exc}")
            set_error(error=f"Speech API error: {exc}")
            if is_transient_error(exc):
                time.sleep(1.0)
                continue
            time.sleep(0.5)
        except (ConnectionResetError, TimeoutError) as exc:
            add_caption(f"[{now_ts()}] Connection issue: {exc}")
            set_error(error=f"Connection issue: {exc}")
            time.sleep(1.0)
            continue
        except OSError as exc:
            add_caption(f"[{now_ts()}] Connection issue: {exc}")
            set_error(error=f"Connection issue: {exc}")
            time.sleep(1.0)
        except Exception as exc:
            set_error(error=f"Unexpected error: {type(exc).__name__}: {exc}")
            time.sleep(0.5)

    with state_lock:
        transcriber_thread = None
        if listener_thread is None:
            recording = False


def clear_audio_queue() -> None:
    while True:
        try:
            audio_q.get_nowait()
        except queue.Empty:
            break


def print_microphones() -> None:
    try:
        names = sr.Microphone.list_microphone_names()
    except Exception as exc:
        print(f"[{now_ts()}] Could not list microphones: {exc}")
        return

    print("Available microphones:")
    for idx, name in enumerate(names):
        print(f"  [{idx}] {name}")
    selected = "default" if MIC_DEVICE_INDEX is None else str(MIC_DEVICE_INDEX)
    print(f"Using microphone device index: {selected}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start_recording():
    global listener_thread, transcriber_thread, recording, latest_error

    with state_lock:
        listener_alive = listener_thread is not None and listener_thread.is_alive()
        transcriber_alive = transcriber_thread is not None and transcriber_thread.is_alive()
        if recording or listener_alive or transcriber_alive:
            return jsonify({"ok": True, "recording": True, "message": "already recording"})
        stop_event.clear()
        clear_audio_queue()
        recording = True
        latest_error = ""
        listener_thread = threading.Thread(target=listener_worker, daemon=True)
        transcriber_thread = threading.Thread(target=transcriber_worker, daemon=True)
        listener_thread.start()
        transcriber_thread.start()
    return jsonify({"ok": True, "recording": True})


@app.route("/api/stop", methods=["POST"])
def stop_recording():
    global listener_thread, transcriber_thread

    stop_event.set()
    with state_lock:
        listener_to_join = listener_thread
        transcriber_to_join = transcriber_thread

    if listener_to_join is not None:
        listener_to_join.join(timeout=4.0)
    if transcriber_to_join is not None:
        transcriber_to_join.join(timeout=4.0)

    mark_recording(False)
    with state_lock:
        listener_thread = None
        transcriber_thread = None
    clear_audio_queue()
    return jsonify({"ok": True, "recording": False})


@app.route("/api/status")
def status():
    with state_lock:
        return jsonify({"recording": recording, "error": latest_error})


@app.route("/api/latest")
def latest():
    with state_lock:
        return jsonify({"recording": recording, "captions": list(captions)})


if __name__ == "__main__":
    print_microphones()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
