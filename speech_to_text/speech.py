import speech_recognition as sr
from datetime import datetime
import time
import socket
import http.client
import urllib.error
from pathlib import Path

def now_ts():
    return datetime.now().strftime("%H:%M:%S")

def log_line(text, transcript_path):
    print(text)
    try:
        with transcript_path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
    except OSError as e:
        print(f"[{now_ts()}] transcript write error: {e}")

def is_transient_request_error(exc):
    transient_types = (
        ConnectionResetError,
        BrokenPipeError,
        TimeoutError,
        socket.timeout,
        socket.gaierror,
        ConnectionError,
        urllib.error.URLError,
        http.client.HTTPException,
    )
    return isinstance(exc, transient_types)

def recognize_with_retry(recognizer, audio, language="en-GB", retries=5, base_delay=0.6):
    for attempt in range(1, retries + 1):
        try:
            return recognizer.recognize_google(audio, language=language)
        except sr.UnknownValueError:
            raise
        except sr.RequestError as e:
            if not is_transient_request_error(e):
                raise
            if attempt == retries:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"[{now_ts()}] API temporarily unavailable "
                f"(attempt {attempt}/{retries}): {e}. Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
        except Exception as e:
            if not is_transient_request_error(e):
                raise
            if attempt == retries:
                raise sr.RequestError(str(e)) from e
            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"[{now_ts()}] Connection issue "
                f"(attempt {attempt}/{retries}): {e}. Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)

def main():
    r = sr.Recognizer()
    r.energy_threshold = 300  # you can tweak
    r.dynamic_energy_threshold = True
    transcript_path = Path(__file__).with_name("transcript.txt")

    mic = sr.Microphone()

    print("SignBridge Speech-to-Text")
    print("Press Ctrl+C to stop.\n")

    with mic as source:
        print("Calibrating mic (1 sec)...")
        r.adjust_for_ambient_noise(source, duration=1)

    while True:
        try:
            with mic as source:
                print("Listening...")
                audio = r.listen(source, phrase_time_limit=6)

            # Google Web Speech API (default in SpeechRecognition) with retries.
            text = recognize_with_retry(r, audio, language="en-GB")

            line = f"[{now_ts()}] {text}"
            log_line(line, transcript_path)

        except sr.UnknownValueError:
            print(f"[{now_ts()}] (couldn't understand)")
        except sr.RequestError as e:
            print(f"[{now_ts()}] API error after retries: {e}")
            time.sleep(1.0)
        except (ConnectionResetError, OSError, TimeoutError) as e:
            print(f"[{now_ts()}] connection dropped: {e}. Recovering...")
            time.sleep(1.0)
        except KeyboardInterrupt:
            print(f"\n[{now_ts()}] stopping speech listener.")
            break
        except Exception as e:
            print(f"[{now_ts()}] unexpected error: {type(e).__name__}: {e}. Continuing...")
            time.sleep(0.5)

if __name__ == "__main__":
    main()
