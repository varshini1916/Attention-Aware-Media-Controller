import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:

            x = int(handLms.landmark[8].x * frame.shape[1])
            y = int(handLms.landmark[8].y * frame.shape[0])

            cv2.circle(frame, (x,y), 8, (0,255,0), -1)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            cv2.line(canvas, (prev_x, prev_y), (x, y), (255,0,0), 5)

            prev_x, prev_y = x, y

    else:
        prev_x, prev_y = 0, 0

    frame = cv2.add(frame, canvas)

    cv2.imshow("Air Drawing", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        canvas = np.zeros_like(frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()