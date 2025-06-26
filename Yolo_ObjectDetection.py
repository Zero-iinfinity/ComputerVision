import cv2 as cv
from ultralytics import YOLO

model = YOLO('yolov8n.pt')


cam = cv.VideoCapture(0)

target_class = ["person", "knife"]

while True:
    ret, frame = cam.read()
    results = model.track(frame, persist = True)
    annotated_frame = frame.copy()
    for box in results[0].boxes:
        #This loops over every detected box in the current frame’s result. Each box has coordinates, confidence, class ID, etc

        class_id = int(box.cls[0].item())
        #Here we extract the class index (e.g., 0 for "person", 2 for "car").

        class_name = model.names[class_id]
        #This converts that class index into a string label by looking it up in model.names.

        if class_name in target_class:
            # annotated_frame = results[0].plot()
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            confidence = box.conf[0].item()

            # Draw bounding box manually
            cv.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(annotated_frame, f'{class_name}: {confidence:.2f}',
                       (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv.imshow('frame', annotated_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


"""
YOLO BOUNDING BOX AND TEXT DRAWING - DETAILED EXPLANATION
========================================================

This section explains the manual drawing of bounding boxes and labels for YOLO detections.
Used when you want to filter specific classes instead of showing all detections.

CONTEXT:
When using results[0].plot(), it draws ALL detected objects regardless of filtering.
Manual drawing gives you complete control over what gets visualized.
"""

# =============================================================================
# LINE 1: EXTRACT BOUNDING BOX COORDINATES
# =============================================================================
# x1, y1, x2, y2 = box.xyxy[0].int().tolist()

"""
BREAKDOWN OF: box.xyxy[0].int().tolist()

1. box.xyxy - Contains bounding box coordinates in "xyxy" format:
   - x1, y1 = top-left corner coordinates
   - x2, y2 = bottom-right corner coordinates
   - Format: [x1, y1, x2, y2]

2. [0] - Gets the first (and usually only) bounding box from this detection
   - Even though there's one box per detection, YOLO returns it as a tensor/array

3. .int() - Converts floating-point coordinates to integers
   - YOLO returns coordinates as floats (e.g., 245.7, 123.4)
   - OpenCV drawing functions need integer pixel coordinates
   - Example: 245.7 becomes 245

4. .tolist() - Converts PyTorch tensor to Python list
   - YOLO uses PyTorch tensors internally
   - We need regular Python integers for OpenCV
   - Final result: [245, 123, 456, 389]

5. Unpacking - Assigns each coordinate to individual variables:
   - x1 = 245 (left edge)
   - y1 = 123 (top edge)
   - x2 = 456 (right edge)
   - y2 = 389 (bottom edge)

COORDINATE SYSTEM:
- Origin (0,0) is at TOP-LEFT of image
- x increases going RIGHT
- y increases going DOWN
- (x1,y1) is top-left corner of bounding box
- (x2,y2) is bottom-right corner of bounding box
"""

# =============================================================================
# LINE 2: EXTRACT CONFIDENCE SCORE
# =============================================================================
# confidence = box.conf[0].item()

"""
BREAKDOWN OF: box.conf[0].item()

1. box.conf - Contains the confidence score(s) for this detection
   - Value between 0.0 and 1.0
   - Higher = more confident the detection is correct
   - Example: 0.8567 means 85.67% confident

2. [0] - Gets the first confidence score
   - Usually only one score per detection
   - Index 0 accesses the first element

3. .item() - Converts single-element tensor to Python float
   - Changes PyTorch tensor to regular Python number
   - Example: tensor(0.8567) becomes 0.8567
   - Required because OpenCV and string formatting need Python types

CONFIDENCE INTERPRETATION:
- 0.9+ : Very high confidence
- 0.7-0.9 : Good confidence
- 0.5-0.7 : Moderate confidence
- Below 0.5 : Low confidence (usually filtered out)
"""

# =============================================================================
# LINE 3: DRAW BOUNDING BOX RECTANGLE
# =============================================================================
# cv.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

"""
BREAKDOWN OF: cv.rectangle() parameters

1. annotated_frame - The image/frame to draw on
   - This is where the rectangle will be drawn
   - Usually a copy of the original frame

2. (x1, y1) - Top-left corner coordinates as tuple
   - Starting point of the rectangle
   - Must be integers

3. (x2, y2) - Bottom-right corner coordinates as tuple
   - Ending point of the rectangle
   - Must be integers

4. (0, 255, 0) - Color in BGR format (Blue, Green, Red):
   - B=0 (Blue channel - minimum)
   - G=255 (Green channel - maximum)
   - R=0 (Red channel - minimum)
   - Result: Bright Green rectangle
   - Common colors:
     * (0, 255, 0) = Green
     * (255, 0, 0) = Blue
     * (0, 0, 255) = Red
     * (255, 255, 0) = Cyan
     * (255, 0, 255) = Magenta
     * (0, 255, 255) = Yellow

5. 2 - Line thickness in pixels
   - Higher number = thicker lines
   - -1 would fill the entire rectangle
   - 1 = thin line, 2 = medium, 3+ = thick

DRAWING BEHAVIOR:
- Draws a hollow rectangle outline
- Rectangle corners are at exact pixel coordinates
- Color is applied to the border only (not filled)
"""

# =============================================================================
# LINE 4: ADD TEXT LABEL
# =============================================================================
# cv.putText(annotated_frame, f'{class_name}: {confidence:.2f}',
#           (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

"""
BREAKDOWN OF: cv.putText() parameters

1. annotated_frame - The image/frame to draw text on
   - Same frame where we drew the rectangle
   - Text will be overlaid on this image

2. f'{class_name}: {confidence:.2f}' - Text string with formatting:
   - class_name - Object name (e.g., "person", "knife")
   - confidence:.2f - Confidence rounded to 2 decimal places
   - f-string formatting for dynamic text
   - Example results:
     * "person: 0.87"
     * "knife: 0.65"
     * "car: 0.92"

3. (x1, y1-10) - Text position coordinates:
   - x1 - Same x-coordinate as box left edge (aligned with rectangle)
   - y1-10 - 10 pixels ABOVE the bounding box top
   - Why y1-10? Prevents text from overlapping the rectangle
   - Text baseline is at this position

4. cv.FONT_HERSHEY_SIMPLEX - Font type
   - Simple, readable font
   - Other options:
     * cv.FONT_HERSHEY_PLAIN
     * cv.FONT_HERSHEY_DUPLEX
     * cv.FONT_HERSHEY_COMPLEX
     * cv.FONT_HERSHEY_TRIPLEX

5. 0.5 - Font scale (size multiplier)
   - 1.0 = normal size
   - 0.5 = half size
   - 2.0 = double size
   - Adjust based on image resolution

6. (0, 255, 0) - Text color (same green as rectangle)
   - BGR format same as rectangle color
   - Keeps visual consistency

7. 2 - Text thickness in pixels
   - 1 = thin text
   - 2 = medium thickness
   - 3+ = bold text
   - Should match or be slightly thinner than rectangle

TEXT POSITIONING EXPLANATION:
- Text baseline is positioned at (x1, y1-10)
- Baseline is the line that letters "sit" on
- Letters with descenders (g, j, p, q, y) extend below baseline
- y1-10 ensures text appears above the bounding box
- Adjust the -10 value if text overlaps or is too far away
"""

# =============================================================================
# VISUAL EXAMPLE WITH SAMPLE DATA
# =============================================================================
"""
EXAMPLE: Person detected with following data:
- Bounding box: top-left (100, 150), bottom-right (300, 400)
- Confidence: 0.8734
- Class: "person"

STEP-BY-STEP EXECUTION:

1. x1, y1, x2, y2 = 100, 150, 300, 400
2. confidence = 0.8734
3. cv.rectangle draws green rectangle from (100, 150) to (300, 400)
4. cv.putText draws "person: 0.87" at position (100, 140)

VISUAL RESULT:
    (100, 140) → "person: 0.87"  ← Text label
    (100, 150) ┌─────────────────┐ ← Top-left corner
               │                 │
               │   DETECTED      │ ← Green rectangle
               │   PERSON        │
               │                 │
               └─────────────────┘ ← Bottom-right corner
                              (300, 400)
"""

# =============================================================================
# WHY MANUAL DRAWING INSTEAD OF results[0].plot()?
# =============================================================================
"""
ADVANTAGES OF MANUAL DRAWING:

1. COMPLETE CONTROL - Only draw what you want to see
   - Filter by class (only show target_class objects)
   - Skip low-confidence detections
   - Custom visualization logic

2. CUSTOM STYLING - Personalize appearance
   - Different colors for different object types
   - Custom text formatting
   - Variable thickness based on confidence

3. PERFORMANCE - Draw only necessary elements
   - Skip processing unwanted detections
   - Faster rendering with fewer objects

4. CONSISTENCY - Matches your filtering logic
   - If you filter detections, visualization should match
   - No confusion between what's detected vs. what's shown

5. DEBUGGING - Better understanding of the data
   - See exactly what coordinates are being used
   - Understand the detection pipeline better

COMPARISON:
- results[0].plot() → Draws ALL detections (no filtering)
- Manual drawing → Draws ONLY what you specify

WHEN TO USE EACH:
- Use results[0].plot() when you want to see everything
- Use manual drawing when you need selective visualization
"""

# =============================================================================
# COMMON CUSTOMIZATIONS
# =============================================================================
"""
COLOR CODING BY CLASS:
colors = {
    'person': (0, 255, 0),    # Green
    'knife': (0, 0, 255),     # Red
    'car': (255, 0, 0),       # Blue
}
color = colors.get(class_name, (255, 255, 255))  # White as default

THICKNESS BY CONFIDENCE:
thickness = int(confidence * 5)  # Higher confidence = thicker lines
thickness = max(1, thickness)    # Minimum thickness of 1

CONDITIONAL TEXT:
if confidence > 0.7:
    label = f'{class_name}: {confidence:.2f}'
else:
    label = f'{class_name}: LOW'

POSITION ADJUSTMENTS:
text_y = y1 - 10 if y1 > 20 else y1 + 20  # Avoid top edge
text_x = max(0, x1)  # Avoid left edge
"""
