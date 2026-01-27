import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from collections import deque



MODEL_PATH = "models/real_fake_model.h5"
model = load_model(MODEL_PATH)
print("Model loaded")



def box_distance(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    return abs(x1 - x2) + abs(y1 - y2) + abs(w1 - w2) + abs(h1 - h2)



cap = cv2.VideoCapture(1)   # change to 0 if webcam doesn't open
kernel = np.ones((5, 5), np.uint8)

LOCK_THRESHOLD = 10
lock_counter = 0
locked_box = None


pred_buffer = deque(maxlen=10)



while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    H, W = frame.shape[:2]
    frame_cx, frame_cy = W // 2, H // 2

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 60, 160)

    clean = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    
    best_cnt = None
    best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2

        center_dist = abs(cx - frame_cx) + abs(cy - frame_cy)
        score = area - center_dist * 4

        if score > best_score:
            best_score = score
            best_cnt = cnt

    
    if best_cnt is not None:
        x, y, w, h = cv2.boundingRect(best_cnt)
        current_box = (x, y, w, h)

        if locked_box is None:
            locked_box = current_box
            lock_counter = 1
        else:
            if box_distance(current_box, locked_box) < 25:
                lock_counter += 1
            else:
                locked_box = current_box
                lock_counter = max(lock_counter - 1, 0)
                pred_buffer.clear()

        
        if lock_counter >= LOCK_THRESHOLD:
            x, y, w, h = locked_box
            roi = frame[y:y+h, x:x+w]

            if roi.size > 0:
                
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)

                symbol = cv2.resize(roi_gray, (64, 64))
                symbol = symbol / 255.0
                symbol = symbol.reshape(1, 64, 64, 1)

                prob = model.predict(symbol, verbose=0)[0][0]
                pred_buffer.append(prob)

                avg_prob = sum(pred_buffer) / len(pred_buffer)

           
                if avg_prob >= 0.35:
                    label = f"REAL ({avg_prob:.2f})"
                    color = (0, 255, 0)
                else:
                    label = f"FAKE ({avg_prob:.2f})"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

                cv2.putText(
                    frame,
                    label,
                    (x, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    "LOCKED",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

                cv2.imshow("Symbol (CNN Input)", symbol[0])

        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Locking {lock_counter}/{LOCK_THRESHOLD}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Detection", frame)
    cv2.imshow("Edge Mask", clean)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
