# SignBridge – Real-Time Communication for Deaf and Hearing Users

## Overview
SignBridge is a real-time AI-powered communication system designed to bridge the gap between deaf and hearing individuals. It enables two-way interaction by converting sign language into text and speech into text instantly, allowing seamless conversations without the need for a human interpreter.

This project was developed during a hackathon and awarded first place for innovation and social impact.

## Features
- Sign-to-Text Translation  
  Detects hand gestures and converts sign language into readable text in real-time.

- Speech-to-Text Conversion  
  Captures spoken language and displays it as text for deaf users.

- Two-Way Communication Interface  
  Combines both features into a single interface for smooth interaction.

- Confidence Filtering  
  Improves prediction accuracy by filtering low-confidence predictions.

- Real-Time Processing  
  Provides fast and responsive interaction using live video and audio input.

## Tech Stack
- Python  
- OpenCV (real-time video processing)  
- MediaPipe (hand tracking and landmark detection)  
- TensorFlow / Keras (LSTM model for gesture recognition)  
- Speech Recognition (audio-to-text processing)  
- NumPy (data handling)  

## How It Works
1. The camera captures hand gestures in real-time.  
2. MediaPipe extracts hand landmarks.  
3. The LSTM model processes sequences of landmarks to predict gestures.  
4. The predicted gesture is converted into text and displayed.  
5. Speech input is captured and converted into text.  
6. Both outputs are displayed in a unified interface.  

## Project Structure
SignBridge/
│── models/              # Trained ML models
│── utils/               # Helper functions
│── sign_detection.py    # Sign language detection logic
│── speech_to_text.py    # Speech recognition module
│── app.py               # Main application
│── requirements.txt     # Dependencies
│── README.md

## How to Run
git clone https://github.com/your-username/signbridge.git  
cd signbridge  
pip install -r requirements.txt  
python app.py  

## My Contribution
- Developed the sign language recognition pipeline  
- Integrated MediaPipe with the LSTM model  
- Implemented real-time prediction and confidence filtering  
- Contributed to system integration and application logic  

## Achievements
- First place in hackathon  
- Recognized for innovation and social impact  
- Delivered a functional prototype within a limited timeframe  

## Future Improvements
- Expand support for a wider range of sign language gestures  
- Improve model accuracy with larger datasets  
- Add multilingual speech recognition  
- Deploy as a web or mobile application  
- Integrate text-to-speech for full accessibility  

## Contact
Zeyad Salama  
Email: ziadm202@gmail.com  
GitHub: https://github.com/zeyad-Salama2  
