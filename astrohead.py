import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
import json
import platform
from typing import Optional
import traceback
# decl glob
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
app = Flask(__name__)
img1 = cv2.imread("/Users/dhruvroongta/Downloads/helmet_redscreen.png")
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
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
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



            nose_x, nose_y = int(nose.x * image.shape[1]), int(nose.y * image.shape[0])
            nose_y = int(0.9 * nose_y)

            crop_size = int(scaled_img1.shape[0]/2)
            print(scaled_img1.shape)
            crop_x1, crop_x2 = max(0, nose_x - crop_size), min(image.shape[1], nose_x + crop_size)
            crop_y1, crop_y2 = max(0, nose_y - crop_size), min(image.shape[0], nose_y + crop_size)

            # Get the corresponding region from img1
            cropped_region = scaled_img1[0:crop_size * 2, 0:crop_size * 2]
 
            red_mask = cropped_region[:, :, 2] == 255
            img_copy[crop_y1:crop_y2, crop_x1:crop_x2][~red_mask] = cropped_region[~red_mask]

        except AttributeError as ae: 
            pass
        except Exception as e:
            print(traceback.format_exc())

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



