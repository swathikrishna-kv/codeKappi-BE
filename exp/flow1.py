import threading
import datetime
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import glob
import time
import math
from tqdm import tqdm
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import requests
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
bicep_curl = True





def sendNotification(userId):
    doc_ref = db.collection('Users').document(userId)
    
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
        print(data)
    else:
        print("not exists")
    url = 'https://us-central1-code-3823a.cloudfunctions.net/pushNotif'
    headers = {'Accept': 'application/json'}  # Set the desired content type in the Accept header
    
    data = {
        'registrationToken': data['token'],
        'title': 'Hey ' + userId,
        'body': 'You have started your workout statistics'
    }
    print(data)
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        response_data = response.text  # Assuming the response is in JSON format
        print(response_data)
    else:
        print(f"Request failed with status code {response.status_code}")

COSINE_THRESHOLD = 0.5

def match(recognizer, feature1, dictionary):
    max_score = 0.0
    sim_user_id = ""
    for user_id, feature2 in zip(dictionary.keys(), dictionary.values()):
        score = recognizer.match(
            feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        if score >= max_score:
            max_score = score
            sim_user_id = user_id
    if max_score < COSINE_THRESHOLD:
        return False, ("", 0.0)
    return True, (sim_user_id, max_score)

def recognize_face(image, face_detector, face_recognizer, file_name=None):
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.shape[0] > 1000:
        image = cv2.resize(image, (0, 0),
                           fx=500 / image.shape[0], fy=500 / image.shape[0])

    height, width, _ = image.shape
    face_detector.setInputSize((width, height))
    try:
        dts = time.time()
        _, faces = face_detector.detect(image)
        if file_name is not None:
            assert len(faces) > 0, f'the file {file_name} has no face'

        faces = faces if faces is not None else []
        features = []
        print(f'time detection  = {time.time() - dts}')
        for face in faces:
            rts = time.time()

            aligned_face = face_recognizer.alignCrop(image, face)
            feat = face_recognizer.feature(aligned_face)
            print(f'time recognition  = {time.time() - rts}')

            features.append(feat)
        return features, faces
    except Exception as e:
        print(e)
        print(file_name)
        return None, None

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


class FreshestFrame(threading.Thread):
    def __init__(self, capture, name='FreshestFrame'):
        self.capture = capture
        assert self.capture.isOpened()
        # this lets the read() method block until there's a new frame
        self.cond = threading.Condition()
        # this allows us to stop the thread gracefully
        self.running = False
        # keeping the newest frame around
        self.frame = None
        # passing a sequence number allows read() to NOT block
        # if the currently available one is exactly the one you ask for
        self.latestnum = 0
        # this is just for demo purposes
        self.callback = None
        super().__init__(name=name)
        self.start()

    def start(self):
        self.running = True
        super().start()

    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.running:
            # block for a fresh frame
            (rv, img) = self.capture.read()
            assert rv
            counter += 1
            # publish the frame
            with self.cond:  # lock the condition for this operation
                self.frame = img if rv else None
                self.latestnum = counter
                self.cond.notify_all()
            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        # with no arguments (wait=True), it always blocks for a fresh frame
        # with wait=False it returns the current frame immediately (polling)
        # with a seqnumber, it blocks until that frame is available (or no wait at all)
        # with a timeout argument, may return an earlier frame;
        #   may even be (0, None) if nothing received yet
        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum + 1
                if seqnumber < 1:
                    seqnumber = 1
                rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
                if not rv:
                    return (self.latestnum, self.frame)
            return (self.latestnum, self.frame)

counter = 0
capture = cv2.VideoCapture(0)

# Create an instance of the FreshestFrame class
cur = "bicep curl"
freshest_frame = FreshestFrame(capture)
flag = 1
text = "unknown"
try:
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        directory = 'data'

    # Init models face detection & recognition
        weights = os.path.join(directory, "models",
                           "face_detection_yunet_2022mar.onnx")
        face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
        face_detector.setScoreThreshold(0.87)

        weights = os.path.join(directory, "models", "face_recognition_sface_2021dec.onnx")
        face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    # Get registered photos and return as npy files
    # File name = id name, embeddings of a photo is the representative for the id
    # If many files have the same name, an average embedding is used
        dictionary = {}
    # the tuple of file types, please ADD MORE if you want
        types = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG')
        files = []
        for a_type in types:
            files.extend(glob.glob(os.path.join(directory, 'images', a_type)))

        files = list(set(files))

        for file in tqdm(files):
            image = cv2.imread(file)
            feats, faces = recognize_face(
                image, face_detector, face_recognizer, file)
            if faces is None:
                continue
            user_id = os.path.splitext(os.path.basename(file))[0]
            dictionary[user_id] = feats[0]

        print(f'there are {len(dictionary)} ids')
        while True:
            start_hand = time.time()
            seq_number, frame = freshest_frame.read()
            if frame is not None:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            fetures, faces = recognize_face(image, face_detector, face_recognizer)
            if faces is None:
                text = "unknown"
                continue
            
            for idx, (face,feature) in enumerate(zip(faces, fetures)):
                result, user = match(face_recognizer, feature, dictionary)
                box = list(map(int, face[:4]))
                color = (0, 255, 0) if result else (0, 0, 255)
                thickness = 2
               

                id_name, score = user if result else (f"unknown_{idx}", 0.0)
                text = "{0} ({1:.2f})".format(id_name, score)
                position = (box[0], box[1] - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                cv2.imshow('MediaPipe Pose', image)
            cur_time = time.time()
            
            if flag == 1 :
                if "known" not in text:
                    with open("log.txt", "a") as f:
                        pname = text.split(" ")[0]
                        print("This is jawaan", pname)
                        f.write(f"Person Started: "+ pname + " | time: " + str(datetime.datetime.now())+"\n")
                        sendNotification(pname)
                        flag=0
                    f.close()
            try:
                
                if cur == "bicep curl":
                    landmarks = results.pose_landmarks.landmark
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    angle = calculate_angle(shoulder, elbow, wrist)
                    cv2.putText(image, str(angle),
                                tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                     )
                    if angle > 160:
                        stage = "down"
                       
                    if angle < 30 and stage =='down':
                        stage="up"
                        counter +=1
                        current_time_stamp = datetime.datetime.now()
                        with open("log.txt", "a") as f:
                            f.write(f"Name: {pname} | Bicep counter: {counter} | time: {current_time_stamp}\n")
                        f.close()
                        print(current_time_stamp)
                        print(counter)
                    cv2.rectangle(image, (20,20), (200,100), (255,106,141), -1)
                    cv2.putText(image, f'Bicep: {counter}', (35, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
                    cv2.imshow('MediaPipe Pose', image)
                    if(counter == 3):
                        counter = 0
                        doc_ref = db.collection("Workouts").document(pname)
                        doc_data = doc_ref.get().to_dict()
                        doc_data['bicepCurl']['isCompleted'] = True
                        doc_ref.set(doc_data)
                        with open("log.txt", "a") as f:
                            f.write(f"Person Completed Bicep Curl: "+ pname + " | time: " + str(datetime.datetime.now())+"\n")
                        f.close()
                        cur = "squat"
                elif cur == "squat":
                    landmarks = results.pose_landmarks.landmark
                
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    angle = calculate_angle(hip, knee, ankle)


                    if angle > 160:
                        stage = "down"
                    if angle < 100 and stage =='down':
                        stage="up"
                        counter +=1
                        current_time_stamp = datetime.datetime.now()
                        with open("log.txt", "a") as f:
                            f.write(f"Name: {pname} | Squat counter: {counter} | time: {current_time_stamp}\n")
                        f.close()
                        print(current_time_stamp)
                        print(counter)
                    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                    cv2.putText(image, f'squat: {counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
                    cv2.imshow('MediaPipe Pose', image)
                    if(counter == 2):
                        counter = 0
                        doc_ref = db.collection("Workouts").document(pname)
                        doc_data = doc_ref.get().to_dict()
                        doc_data['squats']['isCompleted'] = True
                        doc_ref.set(doc_data)
                        with open("log.txt", "a") as f:
                            f.write(f"Person Completed Squats: "+ pname + " | time: " + str(datetime.datetime.now())+"\n")
                            f.write(f"Person Completed Warmup: "+ pname + " | time: " + str(datetime.datetime.now())+"\n") 
                        f.close()
                        cur = "pushup"
                elif cur == "pushup":
                    landmarks = results.pose_landmarks.landmark
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    angle = calculate_angle(shoulder, elbow, wrist)
                    cv2.putText(image, str(angle),
                                tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                     )
                    if angle > 130:
                        stage = "down"
                    if angle < 100 and stage =='down':
                        stage="up"
                        counter +=1
                        current_time_stamp = datetime.datetime.now()
                        with open("log.txt", "a") as f:
                            f.write(f"Name: {pname} | Pushup counter: {counter} | time: {current_time_stamp}\n")
                        f.close()
                        print(current_time_stamp)
                        print(counter)
                    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                    cv2.putText(image, f'pushup: {counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
                    cv2.imshow('MediaPipe Pose', image)
                    if(counter == 2):
                        counter = 0
                        doc_ref = db.collection("Workouts").document(pname)
                        doc_data = doc_ref.get().to_dict()
                        doc_data['pushup']['isCompleted'] = True
                        doc_ref.set(doc_data)
                        with open("log.txt", "a") as f:
                            f.write(f"Person Completed pushups: "+ pname + " | time: " + str(datetime.datetime.now())+"\n")
                            f.write(f"Person Completed Workout: "+ pname + " | time: " + str(datetime.datetime.now())+"\n")
                        f.close()
                        break
            except:
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    
    freshest_frame.release()
    cv2.destroyAllWindows()
