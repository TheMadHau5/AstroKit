import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
import json
import platform
from typing import Optional
import argparse
import imutils
import traceback
# decl glob
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
app = Flask(__name__)
import cv2
import numpy as np
import imutils
def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

# Create new blank 300x300 red image

red = (255, 0, 0)

# Load the image
img1 = cv2.imread("/Users/dhruvroongta/Downloads/hand.png")

# Get the image dimensions
(h, w) = img1.shape[:2]

# Calculate the center point of the image
(cX, cY) = (w // 2, h // 2)

# Rotate the image by -33 degrees around the center of the image
rotated = imutils.rotate_bound(img1, -33)

width,height = 300,300
# re a red canvas with the same dimensions as the rotated image
red_canvas = create_blank(width, height, rgb_color=red)
# red_canvas = np.full_like(rotated, (0, 0, 255), dtype=np.uint8)

# Find the offset to place the rotated image at the center of the red canvas
offset_x = (red_canvas.shape[1] - rotated.shape[1]) // 2
offset_y = (red_canvas.shape[0] - rotated.shape[0]) // 2

# Paste the rotated image onto the red canvas
red_canvas[offset_y:offset_y+rotated.shape[0], offset_x:offset_x+rotated.shape[1]] = rotated

# Save the result
cv2.imwrite("/Users/dhruvroongta/Downloads/hand_rotated_red_background.png", red_canvas)


"""
img1 = cv2.imread("/Users/dhruvroongta/Downloads/hand.png")
(h, w) = img1.shape[:2]
(cX, cY) = (w // 2, h // 2)
# rotate our image by 45 degrees around the center of the image
M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
#rotated = cv2.warpAffine(img1, M, (w, h))
rotated = imutils.rotate_bound(img1, -33)
# cv2.imshow("Rotated by 45 Degrees", rotated)
cv2.imwrite("/Users/dhruvroongta/Downloads/hand2.png", rotated)
"""
"""

camid = -1
if platform.system() == "Darwin":
    camid = 0

cap = cv2.VideoCapture(camid)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_copy = None

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
            distance = np.sqrt(
                ((right_ear.x - left_ear.x) ** 2) + ((right_ear.y - left_ear.y) ** 2))
            optimal_distance = 0.058  # Desired optimal distance
            scaling_factor = distance/optimal_distance  # Calculate the scaling factor
            
            if scaling_factor <= 0:
                scaling_factor = 0.001
            print(scaling_factor)
            # Scale the image1 (img1) based on the calculated scaling factor
            
            img_copy = image.copy()
            # Scale the image1 (img1) based on the calculated scaling factor
            print(img1.shape)
            scaled_width = int(133 * float(scaling_factor))
            scaled_height = int(133 * float(scaling_factor))
            print(scaled_height,scaled_width)
            scaled_img1 = cv2.resize(img1, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)



            wrist_x, wrist_y = int(right_wrist.x * image.shape[1]), int(right_wrist.y * image.shape[0])
            #nose_y = nose_y - scaled

            crop_size = int(scaled_img1.shape[0]/2)
            print(scaled_img1.shape)
            crop_x1, crop_x2 = max(0, wrist_x - crop_size), min(image.shape[1], wrist_x + crop_size)
            crop_y1, crop_y2 = max(0, wrist_y - crop_size), min(image.shape[0], wrist_y + crop_size)

            # Get the corresponding region from img1
            cropped_region = scaled_img1[0:crop_size * 2, 0:crop_size * 2]
 
            red_mask = cropped_region[:, :, 2] == 255
            img_copy[crop_y1:crop_y2, crop_x1:crop_x2][~red_mask] = cropped_region[~red_mask]

        except AttributeError as ae: 
            pass
        except Exception as e:
            print(traceback.format_exc())

        # Render detections
        # if to_run.landmarks_debug:
        #     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
        #                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        #                             )
        # mp_drawing.
        #cv2.imshow('Mediapipe Feed', image)
        try: 
            cv2.imshow('Mediapipe Feed', img_copy)
        except Exception:
            pass

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        ret,buffer = cv2.imencode('.jpg',image)
        frame = buffer.tobytes()
        # yield(b'--frame\r\n'
        #         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()



"""