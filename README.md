# Sign-Language-Interpreter-Project

## Live Speech Captions UI (Flask)

Run a minimal local web UI that starts/stops server-side microphone transcription and shows live captions.

1. Install dependencies:

```bash
pip install flask SpeechRecognition pyaudio
```

2. Start the web app:

```bash
python app/app.py
```

3. Open `http://127.0.0.1:5000` in your browser.

Notes:
- Use **Start** to begin recording and **Stop** to end recording.
- The UI polls `/api/latest` and `/api/status` every 1000ms and updates without page reload.
- Captions are transcribed server-side with Google Speech Recognition (`language="en-GB"`).
- Transcript lines are appended to `speech_to_text/transcript.txt`.
