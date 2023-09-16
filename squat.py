
import datetime
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

squat_counter = 0
squat_status = False
exercise_threshold = 5  

def calculate_angle(a, b, c):
    ab = np.array(b) - np.array(a)
    bc = np.array(c) - np.array(b)
    dot_product = np.dot(ab, bc)
    magnitude_ab = np.linalg.norm(ab)
    magnitude_bc = np.linalg.norm(bc)
    cosine_angle = dot_product / (magnitude_ab * magnitude_bc)
    angle = np.arccos(cosine_angle)
    return angle * 180.0 / np.pi

def detection_body_part(landmarks, body_part_name):
    return [
        landmarks[mp_pose.PoseLandmark[body_part_name].value].x,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].y,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].visibility
    ]

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret,frame = cap.read()
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            left_knee_angle = calculate_angle(
                detection_body_part(landmarks, "LEFT_HIP"),
                detection_body_part(landmarks, "LEFT_KNEE"),
                detection_body_part(landmarks, "LEFT_ANKLE")
            )

            if left_knee_angle > 160:
                squat_status = True
            
            if squat_status and left_knee_angle < 100:
                squat_status = False
                current_time_stamp = datetime.datetime.now()
                # write counter and time stamp to log.txt
                squat_counter += 1
                with open("log.txt", "a") as f:
                    f.write(f"Squat counter: {squat_counter} | time: {current_time_stamp}\n")
                
                f.close()
        
        except:
            pass
        
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        cv2.putText(image, f'Squat: {squat_counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )      
        
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
