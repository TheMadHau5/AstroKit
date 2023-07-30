import cv2
import mediapipe as mp
import numpy as np
from flask import Flask,request,jsonify,render_template,Response
import json
import platform

# decl glob
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
app = Flask(__name__)
last_stance = None
reps = 0
exercise = "latr" # curl, jack, press, latr
landmarks_debug = True


camid = -1

if platform.system() == "Darwin":
    camid = 0

cap = cv2.VideoCapture(camid)

directions = {
    "stance": last_stance,
    "reps": reps,
    "exercise": exercise
}


def calculate_angle(a,b,c):
    radians = np.pi + np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(b[1]-a[1], b[0]-a[0])
    angle = radians*180.0/np.pi

    if angle >180.0:
        angle = angle - 360

    return angle

def get_stance(elbow_angle_l, elbow_angle_r, shoulder_angle_l, shoulder_angle_r):
    stance = None
    if abs(shoulder_angle_l) < 20 and abs(shoulder_angle_r) < 20:
        if abs(elbow_angle_l) > 160 and abs(elbow_angle_r) > 160:
            stance = "stand"
        elif abs(elbow_angle_l) > 160 and abs(elbow_angle_r) < 20:
            stance = "rcurl"
        elif abs(elbow_angle_l) < 20 and abs(elbow_angle_r) > 160:
            stance = "lcurl"
        elif abs(elbow_angle_l) < 20 and abs(elbow_angle_r) < 20:
            stance = "curl"
    elif abs(shoulder_angle_l) > 160 and abs(shoulder_angle_r) > 160 and abs(elbow_angle_l) > 160 and abs(elbow_angle_r) > 160:
        stance = "hands up"
    elif exercise != "jack" and 110 > shoulder_angle_l > 70 and 110 > shoulder_angle_r > 70 and abs(elbow_angle_l) > 140 and abs(elbow_angle_r) > 140:
        stance = "t pose"
    return stance

def count_reps(stance: str):
    global last_stance
    global reps
    if exercise == "jack" and last_stance == "stand" and stance == "hands up":
        reps += 1
    elif exercise == "curl" and last_stance == "stand" and stance == "curl":
        reps += 1
    elif exercise == "press" and last_stance == "hands up" and stance == "curl":
        reps += 1
    elif exercise == "latr" and last_stance == "stand" and stance == "t pose":
        reps += 1

def generate_frames():
    global last_stance
    global reps
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
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                # Calculate angle
                elbow_angle_l = calculate_angle(wrist_l, elbow_l, shoulder_l)
                elbow_angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
                shoulder_angle_l = calculate_angle(elbow_l, shoulder_l, hip_l)
                shoulder_angle_r = calculate_angle(hip_r, shoulder_r, elbow_r)
                # Visualize angle
                print(elbow_angle_l, elbow_angle_r, shoulder_angle_l, shoulder_angle_r)
                stance = get_stance(elbow_angle_l, elbow_angle_r, shoulder_angle_l, shoulder_angle_r)
                print(stance)

                if stance and stance != last_stance:
                    # do thingy
                    count_reps(stance)
                    last_stance = stance
                print(reps)

                directions['stance'] = stance
                directions['reps'] = reps
                directions['exercise'] = exercise
            except AttributeError:
                pass
            except Exception as e:
                print(e)

            

            # Render detections
            if landmarks_debug:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )

            #cv2.imshow('Mediapipe Feed', image)
            ret,buffer = cv2.imencode('.jpg',image)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # ## read the camera frame
        # success,frame=camera.read()
        # if not success:
        #     break
        # else:
        #     ret,buffer=cv2.imencode('.jpg',frame)
        #     frame=buffer.tobytes()

        # yield(b'--frame\r\n'
        #            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/directions')
def directions_func():
    return json.dumps(directions)

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run()#debug=True)
