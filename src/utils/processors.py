import numpy as np
import cv2
from mediapipe.python.solutions.face_detection import *
from enum import Enum

class DETECTOR_NAMES(Enum):
    MEDIAPIPE = "mediapipe"
    HAARCASCADE = "haarcascade"

class preprocessor():
    def __init__(self, detection_model, model_args):
        self.detection_model = detection_model
        self.model_args = model_args
    def __call__(self, image:np.ndarray):
        return self.process(image)
    def process(self, image):
        # crop 
        if self.detection_model == DETECTOR_NAMES.MEDIAPIPE:
            image = mediapipe_crop_image(image,**self.model_args)
        elif self.detection_model == DETECTOR_NAMES.HAARCASCADE:
            image = haarcascade_crop_image(image, **self.model_args)

        return image
    
def get_detection_model_args(detection_model):
    print(detection_model)
    if detection_model == DETECTOR_NAMES.MEDIAPIPE:
        args = {
            "detection_confidence":0.5,
            "model_type":1
        }
    elif detection_model == DETECTOR_NAMES.HAARCASCADE:
        args = {
            "classifer_path":"resources/haarcascade_frontalface_default.xml"
        }
    
    return args

def mediapipe_crop_image(image, detection_confidence, model_type):
    face_detector = FaceDetection(detection_confidence, model_type)
    results = face_detector.process(image)
    faces = results.detections

    if len(faces) == 0:
        print(f"Log: No faces detected returning unchanged image")
        return image
    
    h,w,_ = image.shape # (H,W,C)
    for face in faces:
        bbox = face.location_data.relative_bounding_box
        x,y,w,h = int(bbox.xmin*w), int(bbox.ymin*h), int(bbox.width*w), int(bbox.height*h)
        # image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        image = image[y:y+h, x:x+w]

    return image

def haarcascade_crop_image(image, classifer_path):
    classifier = cv2.CascadeClassifier()

    if not classifier.load(classifer_path):
        print(f"Error: failed to load face classifier")
        return image

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)

    faces = classifier.detectMultiScale(image_gray)
    for face in faces:
        x,y,w,h = face
        # image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        image = image[y:y+h, x:x+w]
    
    return image