import json
import platform
import requests

import cv2
from flask import Flask, request, jsonify, render_template, Response
import mediapipe as mp
import numpy as np

from gesture import *

# decl glob
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
app = Flask(__name__, static_url_path="/static")
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
    def __init__(self, exercise=None):
        self.landmarks_debug = False #TODO Rename to wireframe
        self.set_exercise(exercise)

    def set_exercise(self, exercise):
        self.reps = 0
        self.exercise = exercise
        self.stance = None
        def increment(): self.reps += 1
        self.gman = GestureManager([Gesture(gesture.name, gesture.poses, increment) for gesture in gman.gestures if gesture.name == exercise]) #TODO Allow all

    def update(self, elbow_angle_l, elbow_angle_r, shoulder_angle_l, shoulder_angle_r):
        angles = (elbow_angle_l, elbow_angle_r, shoulder_angle_l, shoulder_angle_r)
        if stand.check(angles):
            self.stance = stand.name
        if curl.check(angles):
            self.stance = curl.name
        if hands_up.check(angles):
            self.stance = hands_up.name
        if t_pose.check(angles):
            self.stance = t_pose.name

        matched = self.gman.match(angles)
        if matched:
            matched.action()


def calculate_angle(a, b, c):
    radians = np.pi + np.arctan2(c.y - b.y, c.x - b.x) - np.arctan2(b.y - a.y, b.x - a.x)
    angle = radians * 180.0 / np.pi
    if angle > 180.0:
        angle = angle - 360
    return angle


def generate_frames():
    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Make detection
            try:
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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

                to_run.update(elbow_angle_l, elbow_angle_r, shoulder_angle_l, shoulder_angle_r)
            except AttributeError: 
                pass
            except Exception as e:
                print(e)

            # Render detections
            if to_run.landmarks_debug:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(192, 128, 64), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(192, 64, 192), thickness=2, circle_radius=2)
                )

            ret, buffer = cv2.imencode(".jpg",frame) #TODO make webp
            frame = buffer.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        cap.release()


to_run = Exercise("curls")

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

@app.route("/update_setting", methods=["POST"])
def update_setting():
    ops = request.get_json(force=True)
    if "exercise" in ops:
        to_run.set_exercise(ops["exercise"])
    if "landmarks_debug" in ops:
        to_run.landmarks_debug = ops["landmarks_debug"]
    return "success"

if __name__ == "__main__":
    app.run()
