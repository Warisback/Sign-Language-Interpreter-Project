import cv2
import os

SIGNS = ["hello", "yes", "no", "help", "please", "thankyou"]
IMAGES_PER_SIGN = 50  # takes about 2 mins total

for sign in SIGNS:
    os.makedirs(f"data/{sign}", exist_ok=True)

cap = cv2.VideoCapture(0)

for sign in SIGNS:
    print(f"\n>>> Get ready to sign: {sign.upper()}")
    print("Press SPACE to start capturing, Q to move to next sign")
    
    count = 0
    capturing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # show status on screen
        cv2.rectangle(frame, (0,0), (640, 50), (10,10,20), -1)
        status = f"CAPTURING {count}/{IMAGES_PER_SIGN}" if capturing else "Press SPACE to start"
        color = (20, 184, 166) if capturing else (255, 255, 255)
        cv2.putText(frame, f"{sign.upper()} — {status}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow("Collecting Data", frame)

        key = cv2.waitKey(10) & 0xFF

        if key == ord(' '):
            capturing = True

        if capturing:
            cv2.imwrite(f"data/{sign}/{count}.jpg", frame)
            count += 1
            if count >= IMAGES_PER_SIGN:
                print(f"Done with {sign}!")
                break

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Data collection complete!")