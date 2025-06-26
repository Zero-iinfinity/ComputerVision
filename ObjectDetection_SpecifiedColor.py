"""object detect with the specified color"""
import cv2 as cv
import numpy as np

# function to convert BGR colors to HSV color and get the upper limit and lowe limit of this HSV(hue,Saturation,Vibrance)
from PIL import Image


def get_limits(color):
    c = np.uint8([[color]])
    #  np.unit8 is use to convet values in to 8-bit unsigned integers &[[is for 3d representation ofccolor]]. Wraps the BGR color as a 1×1 image
    hsvC = cv.cvtColor(c, cv.COLOR_BGR2HSV)  # Converts BGR to HSV

    lowerLimit = hsvC[0][0][0] - 10, 100, 100  # Lower HSV limit (with -10 Hue tolerance)
    upperLimit = hsvC[0][0][0] + 10, 255, 255  # Upper HSV limit (with +10 Hue tolerance)

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)  # Convert to np.uint8 array
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit


# this upper and lower values is like point on circle where arc begins and ends


cam = cv.VideoCapture(0)
yellow = [0, 255, 255]  # yellow in BGR color space
while True:
    ret, frame = cam.read()
    hsvImage = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # converting BGR to HSV because this is the best colour used for color detection task

    lowerLimit, upperLimit = get_limits(color=yellow)

    mask = cv.inRange(hsvImage, lowerLimit, upperLimit)
    # give a mask on the image for which we have given the upper&lower limit . like it covers those porting of image pixel in rage converts to white and not in range to black (range btw upper & lower limit)
    mask_ = Image.fromarray(mask)
    # This line of code is converting a NumPy array called mask into a PIL (Python Imaging Library) Image object.
    bbox = mask_.getbbox()
    print(bbox)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv.imshow("frame", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv.destroyAllWindows()