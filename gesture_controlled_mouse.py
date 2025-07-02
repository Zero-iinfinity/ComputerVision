import mediapipe as mp
import cv2 as cv
import helper_functions

cap = cv.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    success, frame = cap.read()

    results = helper_functions.detect_hands(frame,hands)
    helper_functions.draw_hands(frame,results)
    landmarks = helper_functions.get_landmarks(results,frame)
    print(landmarks)

    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()