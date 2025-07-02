import mediapipe as mp
import cv2 as cv
import helper_functions
import pyautogui
import time
import numpy as np
import autopy
import math


pTime = 0

frame_reduction = 100
smooth_factor = 20
prevX,prevY = 0,0
currX,currY = 0,0

wScr,hScr = pyautogui.size()
wCam, hCam = 640, 480

cap = cv.VideoCapture(0)
cap.set(3,wCam)#for width
cap.set(4,hCam)# for height

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

while True:
    success, frame = cap.read()

    frame = cv.flip(frame, 1) # flip horizontally

    cv.rectangle(frame,(frame_reduction,frame_reduction),(wCam-frame_reduction,hCam-frame_reduction),(255,255,0),2)

    results = helper_functions.detect_hands(frame,hands)
    helper_functions.draw_hands(frame,results)
    lmks = helper_functions.get_landmarks(results,frame)
    # print(landmarks)
    if len(lmks) !=0 :

        x1,y1 = lmks[8][1],lmks[8][2]
        x2, y2 = lmks[12][1], lmks[12][2]

        # cv.circle(frame,(x1,y1),10,(0,255,0),cv.FILLED)
        # cv.circle(frame,(x2,y2),10,(0,255,0),cv.FILLED)

        if lmks[8][2]<lmks[6][2] and lmks[12][2]>lmks[10][2] :
            x3 = np.interp(x1,(frame_reduction,wCam-frame_reduction),(0,wScr))
            y3 = np.interp(y1, (frame_reduction, hCam-frame_reduction), (0, hScr))

            # Clamp to be inside the screen (autopy crashes even at max edge sometimes)
            x3 = max(0, min(x3, wScr - 2))
            y3 = max(0, min(y3, hScr - 2))
            print(f"Moving mouse to: ({x3}, {y3})  |  Screen size: ({wScr}, {hScr})")
            cv.circle(frame, (x1, y1), 10, (255, 0, 0), cv.FILLED)

            #the mose is very shaky, so we have to smoothen its value
            currX = prevX + (x3 - prevX) / smooth_factor
            currY = prevY + (y3 - prevY) / smooth_factor



            # pyautogui.moveTo(x3, y3) ##gives less fps
            autopy.mouse.move(x3, y3) #give more fps
            prevX,prevY = currX,currY

        elif lmks[8][2]<lmks[6][2] and lmks[12][2]<lmks[11][2]:
            distance = math.hypot(x2-x1,y2-y1)
            # print(distance)
            cv.circle(frame,(x1,y1),10,(0,255,0),cv.FILLED)
            cv.circle(frame,(x2,y2),10,(0,255,0),cv.FILLED)
            # cv.putText(frame, "chilling", (400, 70), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

            if distance < 35:
                cv.circle(frame, (x1, y1), 10, (0, 0, 255), cv.FILLED)
                cv.circle(frame, (x2, y2), 10, (0, 0, 255), cv.FILLED)
                cv.putText(frame, "Mouse Click", (400, 70), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
                autopy.mouse.click()

            else:
                cv.putText(frame, "chilling", (400, 70), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(frame,f'FPS:{int(fps)}',(20,70),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),3)

    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


