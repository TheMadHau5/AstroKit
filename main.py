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

camid = -1

if platform.system() == "Darwin":
    camid = 0

cap = cv2.VideoCapture(camid)

directions = {
    "left": "SOMETHING IS FUCKED UP",
    "right": "SOMETHING IS ACTUALLY FUCKED"
}



def calculate_angle(a,b,c):
    radians = np.pi + np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(b[1]-a[1], b[0]-a[0])
    angle = radians*180.0/np.pi

    if angle >180.0:
        angle = angle - 360

    return angle

def get_stance(elbow_angle_l, elbow_angle_r, shoulder_angle_l, shoulder_angle_r):
    stance = None
    if shoulder_angle_l < 20 and shoulder_angle_r < 20:
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
    elif 100 > shoulder_angle_l > 70 and 100 > shoulder_angle_r > 70 and abs(elbow_angle_l) > 160 and abs(elbow_angle_r) > 160:
        stance = "t pose"
    return stance

def arms_facing_direction(shoulder, elbow, wrist):
    shoulder_y, elbow_y, wrist_y = shoulder[1], elbow[1], wrist[1]
    #print(shoulder_y, elbow_y, wrist_y)
    if abs(wrist_y - shoulder_y) < 0.07:
        return "Parallel"
    elif wrist_y < shoulder_y:
        return "Up"
    elif wrist_y > shoulder_y:
        return "Down"

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

                direction_l = arms_facing_direction(shoulder_l, elbow_l, wrist_l)
                direction_r = arms_facing_direction(shoulder_r, elbow_r, wrist_r)
                #print(direction_l, direction_r)
                directions['left'] = direction_l
                directions['right'] = direction_r
                #print("Direction:", direction)
                # Visualize angleS
                cv2.putText(image, str(direction_l) + "Left",
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                cv2.putText(image, str(direction_r) + "Right",
                            tuple(np.multiply(elbow, [640, 580]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
            except:
                pass


            # Render detections
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
