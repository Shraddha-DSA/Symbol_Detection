import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model


MODEL_PATH = "models/real_fake_model.h5"
model = load_model(MODEL_PATH)
print(" Model loaded")



def box_distance(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    return abs(x1-x2) + abs(y1-y2) + abs(w1-w2) + abs(h1-h2)



cap = cv2.VideoCapture(1)   # change to 0 if needed
kernel = np.ones((5, 5), np.uint8)

LOCK_THRESHOLD = 10
lock_counter = 0
locked_box = None
prediction_text = ""


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

   
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)

    mask = cv2.inRange(
        blur,
        np.array([0, 0, 180]),
        np.array([180, 60, 255])
    )

    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_cnt = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000 and area > best_area:
            best_area = area
            best_cnt = cnt

  
    if best_cnt is not None:
        x, y, w, h = cv2.boundingRect(best_cnt)
        current_box = (x, y, w, h)

        if locked_box is None:
            locked_box = current_box
            lock_counter = 1
        else:
            if box_distance(current_box, locked_box) < 20:
                lock_counter += 1
            else:
                locked_box = current_box
                lock_counter = 0
                prediction_text = ""

        
        if lock_counter >= LOCK_THRESHOLD:
            x, y, w, h = locked_box

            roi = frame[y:y+h, x:x+w]

            if roi.size > 0:
                
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                roi_thresh = cv2.adaptiveThreshold(
                    roi_gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    11,
                    2
                )

                kernel_small = np.ones((3, 3), np.uint8)
                roi_clean = cv2.morphologyEx(
                    roi_thresh,
                    cv2.MORPH_OPEN,
                    kernel_small,
                    iterations=1
                )

                symbol = cv2.resize(roi_clean, (64, 64))
                symbol = symbol / 255.0
                symbol = symbol.reshape(1, 64, 64, 1)

               
                prob = model.predict(symbol, verbose=0)[0][0]

                if prob >= 0.5:
                    prediction_text = f"REAL ({prob:.2f})"
                    color = (0, 255, 0)
                else:
                    prediction_text = f"FAKE ({prob:.2f})"
                    color = (0, 0, 255)

               
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

                cv2.putText(
                    frame,
                    prediction_text,
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

                cv2.imshow("Symbol Final", symbol[0])

        
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
    cv2.imshow("Mask", clean)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()
