import cv2 as cv
import numpy as np

'''this code is for reading image using OpenCV'''

import cv2 as cv
print("vikas_1")
image_path = r"C:\Users\Vikas Dubey\Pictures\Screenshots\Screenshot 2025-04-22 014532.png"

image = cv.imread(image_path)

cv.imshow('my image',image)
cv.waitKey(0)

print("vikas")

'''this code is for reading video using OpenCV'''

import cv2 as cv
video_path = r"C:\Users\Vikas Dubey\Pictures\Camera Roll\WIN_20250624_19_22_54_Pro.mp4"
video = cv.VideoCapture(video_path)
ret = True
while ret:
    ret,frame = video.read()
    if ret:
        cv.imshow("frame",frame)
        cv.waitKey(40)
video.release()
cv.destroyAllWindows

'''this is code for opening  the webcam using opencv'''

import cv2 as cv

webcam = cv.VideoCapture(0)

while True:
    ret,frame = webcam.read()

    cv.imshow("frame",frame)
    if cv.waitKey(40) & 0xff == ord('q'):
        break

webcam.release()
cv.destroyAllWindows()

'''this is code for reshaping image using OpenCV'''

import cv2 as cv

image_path = r"C:\Users\Vikas Dubey\Pictures\Screenshots\Screenshot 2025-04-22 014532.png"
image = cv.imread(image_path)

resize_image = cv.resize(image,(225,225)) #this is Width & Height

print(image.shape,resize_image.shape)# this is height & width

cv.imshow('resized_imag', resize_image)
cv.waitKey(0)


'''this is code for Cropping'''

# import cv2 as cv
# image_path = r"C:\Users\Vikas Dubey\Pictures\Screenshots\Screenshot 2025-02-05 012632.png"
# image = cv.imread(image_path)

# croped_image = image[10:640,50:640]

# cv.imshow('my image',croped_image)
# cv.waitKey(0)

'''code to ColourSpaces'''
# OpenCV show our images into BGR color space

# this is all  about chaniging from one color sapce to another

import cv2 as cv

image_path = r"C:\Users\Vikas Dubey\Pictures\Screenshots\Screenshot 2025-02-05 012632.png"

image = cv.imread(image_path)

image_rgb = cv.cvtColor(image,cv.COLOR_BGR2RGB)
image_gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
image_hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
cv.imshow("image",image)
cv.imshow("image_2",image_rgb)
cv.imshow("image_3",image_gray)
cv.imshow("image_4",image_hsv)
cv.waitKey(0)
cv.destroyAllWindows()

'''code to apply blur to an image using openCV'''
'''main use of blur is to remove noise in other words we can say for smoothening '''
# # # blur(),GaussianBlur(),medianBlur(),bilateralFilter()

# evey time we blur a portion of image we are taking avg of the sourrounding pixels and putting in that portion
import cv2 as cv
image_path = r"C:\Users\Vikas Dubey\Pictures\Screenshots\Screenshot 2025-02-05 012632.png"

image = cv.imread(image_path)

kernel_size = 7
#  the kernel size you're passing to cv.GaussianBlur() is either zero or an even number,
# which is invalid. OpenCV requires both width and height of the kernel to be positive and odd integers.

image_blur = cv.blur(image,(kernel_size,kernel_size))#Normalization of pixcel values take place
#same for median blur() above
image_GaussianBlur = cv.GaussianBlur(image,(kernel_size,kernel_size),5)

# # # the 3rd parameter is called sigmaX.  In cv2.GaussianBlur(), sigmaX controls how much the image is blurred along the horizontal (X) axis.
# # # Here’s how it works:
# # # - A small sigmaX (like 0.5 or 1) means the Gaussian kernel is narrow—so only nearby pixels influence the blur. This results in a mild blur.
# # # - A large sigmaX (like 5 or 10) spreads the kernel wider—so more distant pixels contribute, leading to a stronger, smoother blur.

cv.imshow("image",image)
cv.imshow("image_2",image_blur)
cv.imshow("image_3",image_GaussianBlur)
cv.waitKey(0)
cv.destroyAllWindows()
'''Thresholding greyscale image using OpenCV'''
# thresholding is a technique used to convert a grayscale image into a binary image—essentially turning it into black and white based on pixel intensity.

import cv2 as cv
image_path = r"C:\Users\Vikas Dubey\Pictures\Screenshots\Screenshot 2025-02-05 012632.png"

image = cv.imread(image_path)

image_gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

ret,thresh = cv.threshold(image_gray,80,255,cv.THRESH_BINARY)
#This sets all pixels ≥80 to 255 (white), and the rest to 0 (black).

th2 = cv.adaptiveThreshold(image_gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,21,30) # 21 is bolck size and can not be even and must be >1
th3 = cv.adaptiveThreshold(image_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,21,30)
cv.imshow("image",image)
cv.imshow("image_2",thresh)
cv.imshow("image_3",th2)
cv.imshow("image_4",th3)
cv.waitKey(0)

'''Edge Detection using OpenCV'''
#https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
import numpy as np
image_path = r"C:\Users\Vikas Dubey\Downloads\dog.png"

image = cv.imread(image_path)

canny_edge = cv.Canny(image,100,200)
# cv2.Canny() function, the two key values you're referring to are:
# - threshold1 (lower threshold)
# - threshold2 (upper threshold)
# These thresholds are used during the hysteresis stage of the Canny edge detection algorithm:
# - Pixels with gradient intensity > threshold2 are considered strong edges and are definitely kept.
# - Pixels with intensity between threshold1 and threshold2 are considered weak edges—
# they’re only kept if they’re connected to strong edges.
# - Pixels below threshold1 are discarded as noise.
'''we can use dilate and erode to play with the edges i.e to make them thick or thin '''
kernel = np.ones((3, 3), dtype=np.uint8)  # Fixed: parentheses and dtype
canny_edge_d = cv.dilate(canny_edge, kernel)
#same for erode

# Display images
cv.imshow("Original Image", image)
cv.imshow("Canny Edges", canny_edge)
cv.imshow("Dilated Edges", canny_edge_d)
cv.waitKey(0)

'''Drawing shapes on images'''

# import numpy as np

# # 1) Create a blank color image (512×512, black background)
# img = np.zeros((512, 512, 3), dtype=np.uint8)

# # 2) Draw a blue diagonal line
# #    cv.line(image, pt1,    pt2,      color,      thickness,    lineType,     shift)
# cv.line(
#     img,
#     (0, 0),               # pt1: start point (x1,y1)
#     (511, 511),           # pt2: end point   (x2,y2)
#     (255, 0, 0),          # color: BGR tuple
#     5,                    # thickness in pixels
#     cv.LINE_AA,           # lineType: anti-aliased
#     0                     # shift: number of fractional bits in point coordinates
# )

# # 3) Draw a green rectangle
# #    cv.rectangle(image, top_left, bottom_right, color, thickness, lineType, shift)
# cv.rectangle(
#     img,
#     (384, 0),             # top_left:   (x,y)
#     (510, 128),           # bottom_right:(x,y)
#     (0, 255, 0),          # color: BGR
#     3,                    # thickness (-1 for filled)
#     cv.LINE_8,            # lineType
#     0                     # shift
# )

# # 4) Draw a filled red circle
# #    cv.circle(image, center, radius, color, thickness, lineType, shift)
# cv.circle(
#     img,
#     (447, 63),            # center: (x,y)
#     63,                   # radius in pixels
#     (0, 0, 255),          # color: BGR
#     -1,                   # thickness (-1 means fill)
#     cv.LINE_8,            # lineType
#     0                     # shift
# )

# # 5) Put white text on the image
# #    cv.putText(image, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)
# cv.putText(
#     img,
#     'OpenCV',             # text string to draw
#     (10, 500),            # org: bottom-left corner of the text
#     cv.FONT_HERSHEY_SIMPLEX,  # fontFace
#     4,                    # fontScale
#     (255, 255, 255),      # color: BGR
#     2,                    # thickness
#     cv.LINE_AA,           # lineType
#     False                 # bottomLeftOrigin (when True, y-coordinate is top of text)
# )

# # 6) Show and wait
# cv.imshow('Canvas', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

'''Drawing contour around objects in the image'''
https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html

image_path = r"C:\Users\Vikas Dubey\Downloads\dog.png"

image = cv.imread(image_path)
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret,th1 = cv.threshold(image_gray,127,255,cv.THRESH_BINARY_INV)
ret,th2 = cv.threshold(image_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
ret, th3 = cv.threshold(image_gray, 127, 255, 0)
#we have to chosse best threshold for our problem by own

contours_1, hierarchy = cv.findContours(th1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours_2, hierarchy = cv.findContours(th2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours_3, hierarchy = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for cnt in contours_3 :
    if cv.contourArea(cnt) >200:
        cv.drawContours(image,cnt,-1,(0,255,0),1)

    x1,y1,w,h = cv.boundingRect(cnt)
    cv.rectangle(image,(x1,y1),(x1+w,y1+h),(0,255,0),2)

for cnt in contours_1 :
    if cv.contourArea(cnt) >200:
        cv.drawContours(image,cnt,-1,(0,255,0),1)

    x1,y1,w,h = cv.boundingRect(cnt)
    cv.rectangle(image,(x1,y1),(x1+w,y1+h),(0,255,0),2)

# for cnt in contours_2 :
    if cv.contourArea(cnt) >200:
        cv.drawContours(image,cnt,-1,(0,255,0),1)

    x1,y1,w,h = cv.boundingRect(cnt)
    cv.rectangle(image,(x1,y1),(x1+w,y1+h),(0,255,0),2)


cv.imshow("image",image)
cv.imshow("image",image_gray)
cv.imshow("image_1",th1)
cv.imshow("image_2",th2)
cv.imshow("image_3",th3)
cv.waitKey(0)
"""Praticing object detection using live webcam"""
webcam = cv.VideoCapture(0)

while True:
    ret,frame = webcam.read()
    if not ret:
         break
    frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)


    ret, th3 = cv.threshold(frame_gray, 127, 255, 0)
    contours_3, hierarchy = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours_3 :
        if cv.contourArea(cnt) >400:
            cv.drawContours(frame,cnt,-1,(0,255,0),1)

        x1,y1,w,h = cv.boundingRect(cnt)
        cv.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),2)

    cv.imshow("frame_1",frame_gray)
    cv.imshow("frame_2",th3)
    cv.imshow("object_detection",frame)

    if cv.waitKey(40) & 0xFF == ord('q'):
         break


webcam.release()
cv.destroyAllWindows()

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


# this upper and lowwer values is like point on circle where arc bedins and ends


cam = cv.VideoCapture(0)
yellow = [0, 255, 255]  # yellow in BGR color space
while True:
    ret, frame = cam.read()
    hsvImage = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # converting BGR to HSV because this is the best colour used for color detection task

    lowerLimit, upperLimit = get_limits(color=yellow)

    mask = cv.inRange(hsvImage, lowerLimit, upperLimit)
    # give a mask on the image for which we have given the upper&lowwer limit . like it covers those porting of image pixcel in rage converts to white and not in range to black (range btw upper & lower limit)
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

