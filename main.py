from gesture import *

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
# from aiortc import RTCPeerConnection, RTCSessionDescription
import json
# import uuid
# import asyncio
import time
import platform
from typing import Optional
import requests

# decl glob
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
app = Flask(__name__, static_url_path="/static")
img1 = cv2.imread("static/helmet_redscreen.png")
scoreboard_server_addr = "http://127.0.0.1:9000"


camid = -1
if platform.system() == "Darwin":
    camid = 0

cap = cv2.VideoCapture(camid)



stand = Pose("stand", ((160, -160), (160, -160), (-20, 20), (-20, 20)))
curl = Pose("curl", ((-20, 20), (-20, 20), (-20, 20), (-20, 20)))
hands_up = Pose("hands up", ((160, -160), (160, -160), (160, -160), (160, -160)))
t_pose = Pose("t-pose", ((140, -140), (140, -140), (70, 110), (70, 110)))

jacks = Gesture("jacks", [stand, hands_up])
curls = Gesture("curls", [stand, curl])
press = Gesture("press", [hands_up, curl])
latr = Gesture("latr", [stand, t_pose])

gman = GestureManager([jacks, curls, press, latr])



class Exercise:
    def __init__(self, exercise: Optional[str] = None):
        self.score = 0
        self.landmarks_debug = False
        self.helmet = False
        self.set_exercise(exercise)

    def set_exercise(self, exercise):
        self.reps = 0
        self.exercise = exercise
        self.last_stance = None
        self.stance = None
        self.gman = GestureManager([Gesture(gesture.name, gesture.poses, lambda: self.increment()) for gesture in gman.gestures if gesture.name == exercise])
  
    def increment(self):
        self.reps += 1
        self.score += 10

    def update(self, elbow_angle_l, elbow_angle_r, shoulder_angle_l, shoulder_angle_r):
        angles = (elbow_angle_l, elbow_angle_r, shoulder_angle_l, shoulder_angle_r)
        if stand.check(angles):
            self.stance = stand.name
        if curl.check(angles):
            self.stance = curl.name
        if hands_up.check(angles):
            self.stance = hands_up.name
        if t_pose.check(angles) and self.exercise != "jacks":
            self.stance = t_pose.name

        matched = self.gman.match(angles)
        if matched:
            matched.action()

        if self.stance != self.last_stance:
            print(self.stance, "from", self.last_stance)
            print(angles)
            self.last_stance = self.stance

def calculate_angle(a, b, c):
    radians = np.pi + np.arctan2(c.y - b.y, c.x - b.x) - np.arctan2(b.y - a.y, b.x - a.x)
    angle = radians * 180.0 / np.pi
    if angle > 180.0:
        angle = angle - 360
    return angle


to_run = Exercise("curls")


def generate_frames():
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Make detection
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Recolor back to BGR
            image = frame

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                hip_l = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                elbow_l = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                wrist_l = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                hip_r = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                shoulder_r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                elbow_r = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                wrist_r = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                # Calculate angle
                elbow_angle_l = calculate_angle(wrist_l, elbow_l, shoulder_l)
                elbow_angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
                shoulder_angle_l = calculate_angle(elbow_l, shoulder_l, hip_l)
                shoulder_angle_r = calculate_angle(hip_r, shoulder_r, elbow_r)
                angles = (elbow_angle_l, elbow_angle_r, shoulder_angle_l, shoulder_angle_r)

                if to_run.helmet:
                    print()
                    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
                    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
                    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
                    distance = np.sqrt(
                        ((right_ear.x - left_ear.x) ** 2) + ((right_ear.y - left_ear.y) ** 2))
                    optimal_distance = 0.116  # Desired optimal distance
                    scaling_factor = distance/optimal_distance  # Calculate the scaling factor

                    if scaling_factor <= 0:
                        scaling_factor = 0.001
                    img_copy = image.copy()
                    scaled_width = int(133 * float(scaling_factor))
                    scaled_height = int(133 * float(scaling_factor))

                    scaled_img1 = cv2.resize(img1, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)

                    nose_x, nose_y = int(nose.x * image.shape[1]), int(nose.y * image.shape[0])
                    nose_y = int(0.9 * nose_y)

                    crop_size = int(scaled_img1.shape[0]/2)
                    crop_x1, crop_x2 = max(0, nose_x - crop_size), min(image.shape[1], nose_x + crop_size)
                    crop_y1, crop_y2 = max(0, nose_y - crop_size), min(image.shape[0], nose_y + crop_size)
                    cropped_region = scaled_img1[0:crop_size * 2, 0:crop_size * 2]
                    red_mask = cropped_region[:, :, 2] == 255
                    img_copy[crop_y1:crop_y2, crop_x1:crop_x2][~red_mask] = cropped_region[~red_mask]
                    image = img_copy

                to_run.update(elbow_angle_l, elbow_angle_r, shoulder_angle_l, shoulder_angle_r)

            except AttributeError: 
                pass
            except Exception as e:
                print(e)

            # Render detections
            if to_run.landmarks_debug:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )

            # cv2.imshow('Mediapipe Feed', image)
            ret, buffer = cv2.imencode('.jpg',image)
            frame = buffer.tobytes()
            yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()
        cv2.destroyAllWindows()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/tick")
def directions_func():
    return json.dumps({i: to_run.__dict__[i] for i in to_run.__dict__ if i != "gman"})


@app.route("/video")
def video():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route('/change_exercise', methods=['POST'])
def change_exercise():
    to_run = to_run.set_exercise(request.form['exercise'])

@app.route("/update_setting", methods=["POST"])
def update_setting():
    ops = request.get_json(force=True)
    print(ops)
    if "exercise" in ops:
        to_run.set_exercise(ops["exercise"])
    if "landmarks_debug" in ops:
        to_run.landmarks_debug = ops["landmarks_debug"]
    if "helmet" in ops:
        to_run.helmet = ops["helmet"]
    if "reset" in ops:
        to_run.__init__("curls")
    if "score" in ops:
        to_run.score = 0

    return "success"

@app.route("/submit_score", methods=["POST"])
def submit_score():
    data = request.get_json(force=True)
    requests.post(
        f'{scoreboard_server_addr}/add_record',
        json=data
    )
    print("New score pushed!")

for pose in [stand, curl, hands_up, t_pose]:
    pose.show()
for gesture in [jacks, curls, press, latr]:
    gesture.show()

if __name__ == "__main__":
    app.run()

