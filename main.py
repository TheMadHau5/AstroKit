import cv2
import mediapipe as mp
import numpy as np
from flask import Flask,request,jsonify,render_template,Response
import json

# decl glob
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
app = Flask(__name__)
cap = cv2.VideoCapture(-1)

directions = {
    "left": "SOMETHING IS FUCKED UP",
    "right": "SOMETHING IS ACTUALLY FUCKED"
}



def calculate_angle(shoulder,elbow,wrist):
    a = np.array(shoulder) # First
    b = np.array(elbow) # Mid
    c = np.array(wrist) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle   
        
    return angle

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
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                # Calculate angle
                angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
                angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
                # Visualize angle
                #print(angle)
                
                direction_l = arms_facing_direction(shoulder_l, elbow_l, wrist_l)
                direction_r = arms_facing_direction(shoulder_r, elbow_r, wrist_r)
                print(direction_l, direction_r)
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
