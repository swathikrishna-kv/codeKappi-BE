#Not working properly


import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

exercise_flag = "Push-up" 
pushup_counter = 0
pushup_status = False
squat_counter = 0
squat_status = False
bicep_curl_counter = 0
bicep_curl_status = False
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
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            if exercise_flag == "Push-up":
                left_shoulder = detection_body_part(landmarks, "LEFT_SHOULDER")
                left_elbow = detection_body_part(landmarks, "LEFT_ELBOW")
                left_wrist = detection_body_part(landmarks, "LEFT_WRIST")

                # Calculate the angle of the left arm
                left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                # Push-up counter logic
                if not pushup_status and left_arm_angle > 150:
                    pushup_status = True
                elif pushup_status and left_arm_angle < 30:
                    pushup_counter += 1
                    pushup_status = False
                    if pushup_counter >= exercise_threshold:
                        exercise_flag = "Squat"
                        pushup_counter = 0

            elif exercise_flag == "Squat":
                left_hip_angle = calculate_angle(detection_body_part(landmarks, "LEFT_SHOULDER"),
                                                 detection_body_part(landmarks, "LEFT_HIP"),
                                                 detection_body_part(landmarks, "LEFT_KNEE"))
                right_hip_angle = calculate_angle(detection_body_part(landmarks, "RIGHT_SHOULDER"),
                                                  detection_body_part(landmarks, "RIGHT_HIP"),
                                                  detection_body_part(landmarks, "RIGHT_KNEE"))
                if not squat_status and (left_hip_angle < 100 or right_hip_angle < 100):
                    squat_status = True
                elif squat_status and (left_hip_angle > 160 and right_hip_angle > 160):
                    squat_counter += 1
                    squat_status = False
                    if squat_counter >= exercise_threshold:
                        exercise_flag = "Bicep Curl"
                        squat_counter = 0

            elif exercise_flag == "Bicep Curl":
                left_shoulder = detection_body_part(landmarks, "LEFT_SHOULDER")
                left_elbow = detection_body_part(landmarks, "LEFT_ELBOW")
                left_wrist = detection_body_part(landmarks, "LEFT_WRIST")

                # Calculate the angle of the left arm
                left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                # Bicep curl counter logic
                if not bicep_curl_status and left_arm_angle > 150:
                    bicep_curl_status = True
                elif bicep_curl_status and left_arm_angle < 30:
                    bicep_curl_counter += 1
                    bicep_curl_status = False

        except:
            pass

        cv2.rectangle(image, (0, 0), (225, 120), (245, 117, 16), -1)

        cv2.putText(image, 'EXERCISE', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, exercise_flag if exercise_flag else "None",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'PUSH-UP REPS', (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(pushup_counter),
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'SQUAT REPS', (15, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(squat_counter),
                    (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'BICEP CURL REPS', (15, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(bicep_curl_counter),
                    (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
