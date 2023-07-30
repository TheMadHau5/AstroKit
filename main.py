import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
import json
import platform
from typing import Optional
# decl glob
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
app = Flask(__name__)
img1 = cv2.imread("AstroKit/static/helmet_redscreen.png")

camid = -1
if platform.system() == "Darwin":
    camid = 0

cap = cv2.VideoCapture(camid)

class Exercise:
    def __init__(self, exercise:Optional[str] = None,landmarks_debug:Optional[bool] = True):
        self.exercise = exercise
        self.reps = 0
        self.score = 0
        self.last_stance = None
        self.stance = None
        self.landmarks_debug = landmarks_debug
        
    
    def set_exercise(self, exercise):
        self.reps = 0
        self.exercise = exercise
        self.last_stance = None
        self.stance = None
        
    def set_stance(self,elbow_angle_l, elbow_angle_r, shoulder_angle_l, shoulder_angle_r):
        if abs(shoulder_angle_l) < 20 and abs(shoulder_angle_r) < 20:
            if abs(elbow_angle_l) > 160 and abs(elbow_angle_r) > 160:
                self.stance = "stand"
            elif abs(elbow_angle_l) < 20 and abs(elbow_angle_r) < 20:
                self.stance = "curl"
        elif abs(shoulder_angle_l) > 160 and abs(shoulder_angle_r) > 160 and abs(elbow_angle_l) > 160 and abs(elbow_angle_r) > 160:
            self.stance = "hands up"
        elif self.exercise != "jack" and 110 > shoulder_angle_l > 70 and 110 > shoulder_angle_r > 70 and abs(elbow_angle_l) > 140 and abs(elbow_angle_r) > 140:
            self.stance = "t pose"
        

    def count_reps(self):
        if self.stance and self.stance != self.last_stance:
            if self.exercise == "jack" and self.last_stance == "stand" and self.stance == "hands up":
                self.reps += 1
                self.score += 10
            elif self.exercise == "curl" and self.last_stance == "stand" and self.stance == "curl":
                self.reps += 1
                self.score += 10
            elif self.exercise == "press" and self.last_stance == "hands up" and self.stance == "curl":
                self.reps += 1
                self.score += 10
            elif self.exercise == "latr" and self.last_stance == "stand" and self.stance == "t pose":
                self.reps += 1
                self.score += 10
        self.last_stance = self.stance

def calculate_angle(a,b,c):
    radians = np.pi + np.arctan2(c.y-b.y, c.x-b.x) - np.arctan2(b.y-a.y, b.x-a.x)
    angle = radians*180.0/np.pi
    if angle >180.0:
        angle = angle - 360
    return angle

to_run = Exercise("curl")

def generate_frames():
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

                elbow_angle_l = calculate_angle(wrist_l, elbow_l, shoulder_l)
                elbow_angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
                shoulder_angle_l = calculate_angle(elbow_l, shoulder_l, hip_l)
                shoulder_angle_r = calculate_angle(hip_r, shoulder_r, elbow_r)
                
                #Start of Amongus Mask
                nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
                left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
                right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
                distance = np.sqrt(
                    ((right_ear.x - left_ear.x) ** 2) + ((right_ear.y - left_ear.y) ** 2))
                optimal_distance = 0.058  # Desired optimal distance
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
                print(scaled_img1.shape)
                crop_x1, crop_x2 = max(0, nose_x - crop_size), min(image.shape[1], nose_x + crop_size)
                crop_y1, crop_y2 = max(0, nose_y - crop_size), min(image.shape[0], nose_y + crop_size)
                cropped_region = scaled_img1[0:crop_size * 2, 0:crop_size * 2]
                red_mask = cropped_region[:, :, 2] == 255
                img_copy[crop_y1:crop_y2, crop_x1:crop_x2][~red_mask] = cropped_region[~red_mask]
                image = img_copy
                #End of Amongus Mask

                print(elbow_angle_l, elbow_angle_r, shoulder_angle_l, shoulder_angle_r)
                to_run.set_stance(elbow_angle_l, elbow_angle_r, shoulder_angle_l, shoulder_angle_r)
                to_run.count_reps()

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

            #cv2.imshow('Mediapipe Feed', image)
            ret,buffer = cv2.imencode('.jpg',image)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()
        cv2.destroyAllWindows()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/tick")
def directions_func():
    return json.dumps(to_run.__dict__)


@app.route("/video")
def video():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route('/change_exercise', methods=['POST'])
def change_exercise():
    to_run = to_run.set_exercise(request.form['exercise'])

if __name__ == "__main__":
    app.run(debug = True)  # debug=True)

