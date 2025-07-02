import cv2 as cv
import time
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands()# ctrl + left click to open the code.
mpDraw = mp.solutions.drawing_utils
cap = cv.VideoCapture(0)

pTime = 0
cTime = 0

while True:
    ret,frame = cap.read()



    imgRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)#this hand module only uses RGB images
    results = hands.process(imgRGB) # this processes the img and gives the result

    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                lmks = []
                h, w, c = imgRGB.shape
                cx = int(lm.x*w)
                cy = int(lm.y*h)

                # print(id,cx,cy)# this is giving values on decimals .so basically they are giving a ratio of image,so for getting it in pixel, we have to multiply it with width, height of our frame

                mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS)# handsLms provides points to place dots and mphands.HAND_CONNECTION connects those points, basically it takes the image/frame on which the coordinates to be drawn and the coordinates of points

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(frame,str(int(fps)),(10,70),cv.FONT_ITALIC,3,[255,0,225],3)

    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()