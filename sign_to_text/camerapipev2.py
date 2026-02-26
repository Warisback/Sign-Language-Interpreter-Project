import cv2
import tensorflow as tf
import numpy as np
import json

model = tf.keras.models.load_model("signbridge_model.h5")
with open("labels.json") as f:
    labels = json.load(f)

IMG_SIZE = 224
current_sign = ""
current_conf = 0.0

cap = cv2.VideoCapture(0)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 10 == 0:
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)
        idx = np.argmax(pred)
        conf = pred[0][idx]

        if conf > 0.75:
            current_sign = labels[str(idx)].upper()
            current_conf = conf
        else:
            current_sign = "..."
            current_conf = 0.0

    # UI overlay
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h-120), (w, h), (10,10,20), -1)
    cv2.putText(frame, current_sign, (20, h-55),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (20,184,166), 4)

    if current_conf > 0:
        bar_w = int((w-40) * current_conf)
        cv2.rectangle(frame, (20, h-25), (20+bar_w, h-10), (20,184,166), -1)
        cv2.rectangle(frame, (20, h-25), (w-20, h-10), (60,60,60), 2)
        cv2.putText(frame, f"{current_conf*100:.0f}%", (w-70, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    cv2.rectangle(frame, (0,0), (w,40), (10,10,20), -1)
    cv2.putText(frame, "SignBridge", (10,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("SignBridge", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Time breakdown for you:

- **Data collection** — 2 mins per sign × 6 signs = **12 minutes**
- **Training** — **~3-5 minutes** on CPU
- **Done** — working real-time detector

## Tips for good accuracy:
- Film in the **same lighting** you'll demo in
- Keep your hand **centred** in frame when collecting
- Do signs slightly differently each time — vary the angle a tiny bit — so the model generalises
- 50 images per sign is enough. Don't waste time doing more

Just install first:
```
pip install tensorflow opencv-python numpy