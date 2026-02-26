
#pip install torch torchvision transformers pillow opencv-python flask flask-socketio
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
#pip install torch torchvision transformers pillow opencv-python flask flask-socketio
# ── LOAD PRETRAINED MODEL FROM HUGGINGFACE ───────────────────────────────────
print("Loading model...")
model_name = "dima806/sign-language-image-classification"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()
print("Model ready!")

# Easy words we care about — filter to these only
TARGET_SIGNS = ["hello", "yes", "no", "help", "please", "thanks", "sorry", "name", "where", "understand"]

def predict_sign(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = extractor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top_prob, top_idx = probs.topk(1)
    label = model.config.id2label[top_idx.item()].lower()
    confidence = top_prob.item()
    return label, confidence

# ── WEBCAM LOOP ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)  # 0 = laptop webcam

current_sign = ""
current_conf = 0.0
frame_count = 0

print("Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run prediction every 10 frames (not every frame — keeps it smooth)
    if frame_count % 10 == 0:
        label, confidence = predict_sign(frame)
        if confidence > 0.7:  # only show if confident
            current_sign = label.upper()
            current_conf = confidence
        else:
            current_sign = "..."
            current_conf = 0.0

    # ── DRAW UI OVERLAY ──────────────────────────────────────────────────────
    h, w = frame.shape[:2]

    # Dark overlay at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 120), (w, h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Sign prediction text
    cv2.putText(frame, current_sign, (20, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (20, 184, 166), 4)  # teal

    # Confidence bar
    if current_conf > 0:
        bar_w = int((w - 40) * current_conf)
        cv2.rectangle(frame, (20, h - 25), (20 + bar_w, h - 10), (20, 184, 166), -1)
        cv2.rectangle(frame, (20, h - 25), (w - 20, h - 10), (60, 60, 60), 2)
        cv2.putText(frame, f"{current_conf*100:.0f}%", (w - 70, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 40), (10, 10, 20), -1)
    cv2.putText(frame, "SignBridge - Sign Detection", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("SignBridge", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()