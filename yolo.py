from ultralytics import YOLO
import threading
import datetime
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
bicep_curl = True

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
camera="http://192.168.1.146:4747/video"
capture = cv2.VideoCapture(camera)
model = YOLO("yolo-Weights/yolov8n.pt")
classNames = ["person"]
random_names = ["Madhav", "Alphin", "Jacob"]
random_age = ["22","24","25"]
random_calory = ["126kcal","204kcal","486kcal"]
# Create an instance of the FreshestFrame class
freshest_frame = FreshestFrame(capture)
counter = 0


try:
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
        # Read the latest frame (blocking until a new frame is available)
            seq_number, frame = freshest_frame.read()

            results = model(frame,stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1,y1,x2,y2 = box.xyxy[0]
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                    if (int(box.cls[0])==0):
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
                        org = [x1,y1]
                        # in filled recatngle with white background and black text fill random name, age and calory
                        if x1 < 320:
                            counter = 0
                        else:
                            counter = 1
                        cv2.rectangle(frame,(x1,y1-90),(x1+150,y1-20),(255,255,255),-1)
                        cv2.putText(frame,"name: "+random_names[counter],(x1,y1-70),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                        
                        cv2.putText(frame,"age: "+random_age[counter],(x1,y1-50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                        
                        cv2.putText(frame,"CAL: "+random_calory[counter],(x1,y1-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)

                        

            cv2.imshow('iamge', frame)
            
            

        # Break the loop if the user presses 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    # Release resources and close the OpenCV window
    freshest_frame.release()
    cv2.destroyAllWindows()
