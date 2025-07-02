import mediapipe as mp
import cv2 as cv




def detect_hands(frame, hands):  # take hands as argument
    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    return results



def draw_hands(frame,results) :
    mp_draw = mp.solutions.drawing_utils

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,handLms, mp.solutions.hands.HAND_CONNECTIONS)

def get_landmarks(results,frame) :
    lm_list = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = frame.shape
                cx = int(lm.x * w)
                cy = int(lm.y * h)

                lm_list.append([id, cx, cy])
    return lm_list

def get_both_landmarks(results, frame):
    all_lmks = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            h, w, _ = frame.shape
            for id, lm in enumerate(handLms.landmark):
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                lm_list.append([id, cx, cy])
            all_lmks.append(lm_list)
    return all_lmks
