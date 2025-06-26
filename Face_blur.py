import cv2 as cv
import mediapipe as mp

cam = cv.VideoCapture(0)



mp_face_detection = mp.solutions.face_detection

#this with....as statement is a modern way of saying after when I am finish do the cleaning
##Old way of doing this
# face_detection = mp_face_detection.FaceDetection(model_selection=0)
# face_detection.start()  # Setup
# results = face_detection.process(image)  # Use it
# face_detection.close()
while True:
    ret, frame = cam.read()
    if not ret:
        break
    H, W, _ = frame.shape
    with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as face_detection :
        #select model_selection  = 1 for long range detection

        img_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        out = face_detection.process(img_rgb)
        # this code gives the location of the face it detected from frame or image

        if out.detections is not None :
            for detection in out.detections:
                location_data = detection.location_data
                bbox = location_data.relative_bounding_box

                x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

                x1 = int(x1 * W)
                y1 = int(y1 * H)
                w = int(w * W)
                h = int(h * H)

                frame[y1:y1+h, x1:x1+w] = cv.blur(frame[y1:y1+h, x1:x1+w], (30,30))

    cv.imshow("frame", frame)
    if cv.waitKey(40) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()




#
# # REAL-TIME FACE BLUR APPLICATION - DETAILED EXPLANATION
# ==============================
#
# ## OVERVIEW
# This explanation covers a real-time face blur application that uses MediaPipe and OpenCV to detect faces in webcam feed and applies blur effects for privacy protection.
#
# ## FEATURES
# - Real-time face detection using MediaPipe
# - Automatic blur effect on detected faces
# - Live webcam feed processing
# - Press 'q' to quit
#
# ## DEPENDENCIES
# - opencv-python (cv2): For video capture and image processing
# - mediapipe: Google's ML framework for face detection
# - Python 3.10 or lower (MediaPipe compatibility)
#
# ## SETUP REQUIREMENTS
# 1. Create conda environment: `conda create -n cv_env python=3.10`
# 2. Install packages: `pip install mediapipe opencv-python numpy`
# 3. Webcam must be connected and working
#
# ## USAGE
# - Run the script
# - Point your face toward the camera
# - Face will be automatically blurred
# - Press 'q' to exit
#
# ---
#
# ## STEP-BY-STEP BREAKDOWN
#
# ### STEP 1: INITIALIZE VIDEO CAPTURE
#
# `cv.VideoCapture(0)` creates a video capture object:
# - Parameter 0: Default camera (usually built-in webcam)
# - Parameter 1: External USB camera (if available)
# - Returns: VideoCapture object for reading frames
#
# ### STEP 2: INITIALIZE MEDIAPIPE FACE DETECTION
#
# `mp.solutions.face_detection` - MediaPipe's pre-trained face detection model:
# - Uses machine learning to detect human faces in images
# - Provides bounding box coordinates for detected faces
# - Returns confidence scores for each detection
#
# ### CONTEXT MANAGER EXPLANATION
#
# The `with...as` statement is a **CONTEXT MANAGER**:
# - Automatically handles setup and cleanup of resources
# - Ensures proper memory management
# - Prevents resource leaks
#
# **OLD WAY (Manual):**
# ```
# face_detection = mp_face_detection.FaceDetection(model_selection=0)
# face_detection.start()  # Setup
# results = face_detection.process(image)  # Use it
# face_detection.close()  # Cleanup (easy to forget!)
# ```
#
# **NEW WAY (Automatic):**
# ```
# with mp_face_detection.FaceDetection() as face_detection:
#     # Use the detector
#     # Cleanup happens automatically when exiting the 'with' block
# ```
#
# ### STEP 3: MAIN PROCESSING LOOP
#
# **INFINITE LOOP** for continuous video processing:
# - Captures frames from webcam
# - Processes each frame for face detection
# - Applies blur effect to detected faces
# - Displays the result
# - Continues until user presses 'q'
#
# #### STEP 3.1: CAPTURE FRAME FROM WEBCAM
#
# `cam.read()` returns **TWO values**:
# - `ret` (boolean): True if frame captured successfully, False if failed
# - `frame` (numpy array): The actual image data as a 3D array [height, width, channels]
#
# **Frame format:** BGR (Blue-Green-Red) color space
# **Frame shape:** (height, width, 3) - 3 channels for BGR
#
# #### STEP 3.2: SAFETY CHECK
#
# Check if frame was captured successfully:
# - If webcam disconnected: ret = False
# - If camera error: ret = False
# - If successful: ret = True
#
# Break the loop if capture fails to prevent infinite loop.
#
# #### STEP 3.3: GET FRAME DIMENSIONS
#
# `frame.shape` returns (height, width, channels):
# - H: Height in pixels (e.g., 480)
# - W: Width in pixels (e.g., 640)
# - _: Channels (3 for BGR), using _ because we don't need it
#
# These dimensions are needed to convert relative coordinates to pixel coordinates.
#
# #### STEP 3.4: FACE DETECTION PROCESSING
#
# **FaceDetection Parameters:**
# - `model_selection=0`: Short-range model (0-2 meters, faster)
# - `model_selection=1`: Full-range model (0-5 meters, slower)
# - `min_detection_confidence=0.5`: Minimum 50% confidence to consider as face
#   - Lower values (0.1): More detections, more false positives
#   - Higher values (0.9): Fewer detections, more accurate
#
# ##### STEP 3.4.1: COLOR SPACE CONVERSION
#
# MediaPipe expects RGB color format, but OpenCV uses BGR:
# - **OpenCV default:** BGR (Blue-Green-Red)
# - **MediaPipe expects:** RGB (Red-Green-Blue)
#
# `cv.cvtColor()` converts between color spaces:
# - `cv.COLOR_BGR2RGB`: Converts from BGR to RGB
# - This is essential for MediaPipe to work correctly
#
# ##### STEP 3.4.2: FACE DETECTION PROCESSING
#
# `face_detection.process(img_rgb)`:
# - Runs the AI model on the RGB image
# - Returns detection results object
# - Contains: bounding boxes, confidence scores, keypoints
#
# **Results structure:**
# - `out.detections`: List of detected faces (None if no faces found)
# - Each detection contains: location_data, score, keypoints
#
# ##### STEP 3.4.3: PROCESS DETECTION RESULTS - DETAILED EXPLANATION OF `out.detections`
#
# ## **COMPREHENSIVE BREAKDOWN OF `if out.detections is not None:`**
#
# ### **What is `out.detections`?**
#
# `out.detections` is the core result returned by MediaPipe's face detection model. Here's what it contains:
#
# **When NO faces are detected:**
# - `out.detections = None`
# - The condition `if out.detections is not None:` evaluates to `False`
# - No face processing occurs, frame displays unchanged
#
# **When faces ARE detected:**
# - `out.detections = [detection1, detection2, ...]` (a Python list)
# - Each `detection` object represents one detected face
# - The condition `if out.detections is not None:` evaluates to `True`
# - Face processing begins
#
# ### **Structure of Each Detection Object**
#
# Each `detection` in the `out.detections` list contains:
#
# 1. **Location Data (`detection.location_data`)**
#    - Spatial information about where the face is located
#    - Contains the bounding box coordinates
#    - Includes facial keypoints (nose tip, eye centers, etc.)
#
# 2. **Confidence Score (`detection.score`)**
#    - Float value between 0.0 and 1.0
#    - Represents how confident the model is that this is actually a face
#    - Higher values = more confident detection
#    - Filtered by `min_detection_confidence` parameter
#
# 3. **Classification Data (if available)**
#    - Additional metadata about the detection
#    - May include face orientation information
#
# ### **Why Check `if out.detections is not None:`?**
#
# This check is **crucial** for several reasons:
#
# 1. **Prevents Crashes:**
#    - If no faces are detected, `out.detections` is `None`
#    - Trying to iterate over `None` would cause a `TypeError`
#    - The check prevents: `for detection in None:` which would crash
#
# 2. **Performance Optimization:**
#    - Skip processing when no faces are present
#    - No unnecessary computation on empty results
#    - Faster frame rate when no faces in view
#
# 3. **Clean Code Logic:**
#    - Clear separation between "faces found" and "no faces" scenarios
#    - Makes the code flow more readable and logical
#
# ### **Alternative Ways to Handle This Check**
#
# **Method 1 (Current approach):**
# ```python
# if out.detections is not None:
#     for detection in out.detections:
#         # Process each face
# ```
#
# **Method 2 (Using empty list default):**
# ```python
# detections = out.detections or []
# for detection in detections:
#     # Process each face
# ```
#
# **Method 3 (Direct boolean check):**
# ```python
# if out.detections:  # Pythonic way - None is falsy
#     for detection in out.detections:
#         # Process each face
# ```
#
# ### **What Happens Inside the Loop**
#
# When `out.detections` contains faces, the loop `for detection in out.detections:` processes each detected face individually:
#
# - **Multiple faces:** If 3 faces are detected, the loop runs 3 times
# - **Single face:** If 1 face is detected, the loop runs 1 time
# - **Each iteration:** One `detection` object is processed
# - **Independent processing:** Each face gets its own bounding box calculation and blur effect
#
# #### STEP 3.4.4: PROCESS EACH DETECTED FACE
#
# Loop through each detected face:
# - Multiple faces can be detected in one frame
# - Each face gets processed separately
# - Each face gets its own bounding box and blur effect
#
# #### STEP 3.4.5: EXTRACT BOUNDING BOX DATA
#
# **Detection structure:**
# - `detection.location_data`: Contains spatial information
# - `location_data.relative_bounding_box`: Normalized coordinates (0-1)
#
# **Relative coordinates** are normalized to image size:
# - 0.0 = left/top edge of image
# - 1.0 = right/bottom edge of image
# - 0.5 = center of image
#
# #### STEP 3.4.6: EXTRACT NORMALIZED COORDINATES
#
# **Bounding box coordinates (all values 0-1):**
# - `xmin`: Left edge of face (0 = left side of image)
# - `ymin`: Top edge of face (0 = top of image)
# - `width`: Width of bounding box
# - `height`: Height of bounding box
#
# **Example:** xmin=0.3, ymin=0.2, width=0.4, height=0.5
# Face is 30% from left, 20% from top, 40% of image width, 50% of image height
#
# #### STEP 3.4.7: CONVERT TO PIXEL COORDINATES
#
# Convert normalized coordinates (0-1) to actual pixel positions:
# - `x1 * W`: Convert relative x to pixel x
# - `y1 * H`: Convert relative y to pixel y
# - `w * W`: Convert relative width to pixel width
# - `h * H`: Convert relative height to pixel height
#
# **Example with 640x480 image:**
# - x1=0.3 × 640 = 192 pixels from left
# - y1=0.2 × 480 = 96 pixels from top
# - w=0.4 × 640 = 256 pixels wide
# - h=0.5 × 480 = 240 pixels tall
#
# #### STEP 3.4.8: APPLY BLUR EFFECT
#
# **Array slicing** to select face region:
# - `frame[y1:y1+h, x1:x1+w]`: Selects rectangular face region
# - `y1:y1+h`: Rows from y1 to y1+height
# - `x1:x1+w`: Columns from x1 to x1+width
#
# **cv.blur() parameters:**
# - `src`: Source image region (the face)
# - `ksize`: Kernel size (30,30) - larger = more blur
# - `(30,30)`: 30×30 pixel averaging kernel
#
# **Assignment back to original frame:**
# - `frame[y1:y1+h, x1:x1+w] = blurred_region`
# - Replaces original face pixels with blurred pixels
#
# ### STEP 3.5: DISPLAY PROCESSED FRAME
#
# `cv.imshow()` displays the processed frame:
# - "frame": Window name/title
# - frame: The processed image with blurred faces
#
# Creates a window showing the live video feed with face blur effect.
#
# ### STEP 3.6: CHECK FOR EXIT CONDITION
#
# `cv.waitKey()` waits for keyboard input:
# - 40: Wait 40 milliseconds (controls frame rate ~25 FPS)
# - `& 0xFF`: Bitwise operation to get the last 8 bits
# - `ord('q')`: ASCII value of 'q' character
#
# If 'q' is pressed, break the loop and end the program.
#
# **Frame rate calculation:**
# - waitKey(40) = 40ms delay = 1000ms/40ms = 25 FPS
# - waitKey(30) = 30ms delay = 33.3 FPS
# - waitKey(1) = 1ms delay = ~1000 FPS (limited by processing speed)
#
# ### STEP 4: CLEANUP AND RESOURCE RELEASE
#
# Proper cleanup is essential to prevent resource leaks:
#
# **cam.release():**
# - Releases the webcam resource
# - Allows other applications to use the camera
# - Prevents "camera in use" errors
#
# **cv.destroyAllWindows():**
# - Closes all OpenCV windows
# - Frees up memory allocated for windows
# - Prevents zombie windows from staying open
#
# ---
#
# ## TROUBLESHOOTING GUIDE
#
# ### COMMON ISSUES AND SOLUTIONS
#
# 1. **"Camera not found" or black screen:**
#    - Try `cam = cv.VideoCapture(1)` for external camera
#    - Check if camera is being used by another app
#    - Verify camera permissions in system settings
#
# 2. **"ModuleNotFoundError: No module named 'mediapipe'":**
#    - Install with: `pip install mediapipe`
#    - Ensure you're in the correct conda environment
#    - Try: `conda activate cv_env`
#
# 3. **Poor face detection performance:**
#    - Increase min_detection_confidence (0.7-0.9)
#    - Ensure good lighting conditions
#    - Try model_selection=1 for longer range
#
# 4. **Lag or slow performance:**
#    - Reduce blur kernel size: (15,15) instead of (30,30)
#    - Increase waitKey delay: 50 instead of 40
#    - Use model_selection=0 for faster processing
#
# 5. **Face detection not working:**
#    - Check if face is clearly visible
#    - Ensure proper lighting
#    - Try different angles
#    - Lower min_detection_confidence to 0.3
#
# 6. **Program won't exit:**
#    - Make sure to press 'q' key while video window is active
#    - Click on video window first, then press 'q'
#    - Use Ctrl+C in terminal as backup
#
# ---
#
# ## PERFORMANCE OPTIMIZATION TIPS
#
# ### OPTIMIZATION STRATEGIES
#
# 1. **Frame Rate Control:**
#    - Current: 25 FPS (waitKey(40))
#    - For smoother video: waitKey(30) = 33 FPS
#    - For better performance: waitKey(50) = 20 FPS
#
# 2. **Blur Effect Tuning:**
#    - Light blur: (15,15) - faster processing
#    - Medium blur: (30,30) - current setting
#    - Heavy blur: (50,50) - slower but more privacy
#
# 3. **Detection Sensitivity:**
#    - Sensitive: min_detection_confidence=0.3
#    - Balanced: min_detection_confidence=0.5 (current)
#    - Conservative: min_detection_confidence=0.7
#
# 4. **Model Selection:**
#    - Fast: model_selection=0 (0-2 meters)
#    - Accurate: model_selection=1 (0-5 meters)
#
# 5. **Memory Management:**
#    - The 'with' statement automatically handles cleanup
#    - No manual memory management needed
#    - MediaPipe handles model loading/unloading
#
# ---
#
# ## EXTENSION IDEAS
#
# ### POSSIBLE ENHANCEMENTS
#
# 1. **Multiple Blur Types:**
#    - Gaussian blur: cv.GaussianBlur()
#    - Motion blur: custom kernel
#    - Pixelation effect: resize down then up
#
# 2. **Face Recognition:**
#    - Save known faces to skip blurring
#    - Blur only unknown faces
#    - Use face_recognition library
#
# 3. **Recording Functionality:**
#    - Save blurred video to file
#    - Screenshot with 's' key
#    - Time-lapse recording
#
# 4. **UI Improvements:**
#    - Add text overlay with instructions
#    - Show detection confidence
#    - FPS counter display
#
# 5. **Advanced Features:**
#    - Emotion detection
#    - Age/gender detection
#    - Multiple face tracking
#    - Background blur instead of face blur