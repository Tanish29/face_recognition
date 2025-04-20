import numpy as np
import cv2
from mediapipe.python.solutions.face_detection import *
from enum import Enum
import torch 
import yaml

with open("configs/config.yml", "r") as file: # load config file
    config = yaml.safe_load(file)  
    device = config['device']

normalisation_mean = (0.485, 0.456, 0.406)
normalisation_std = (0.229, 0.224, 0.225)

class DETECTOR_NAMES(Enum):
    MEDIAPIPE = "mediapipe"
    HAARCASCADE = "haarcascade"

class PreProcessor():
    def __init__(self, detection_model, resize_res):
        self.detection_model = detection_model
        self.resize_res = resize_res
    def __call__(self, image:np.ndarray):
        return self.process(image)
    def process(self, image):
        # crop 
        if self.detection_model == DETECTOR_NAMES.MEDIAPIPE:
            confidence = 0.5
            model_type = 1
            image = mediapipe_crop_image(image,confidence,model_type)
        elif self.detection_model == DETECTOR_NAMES.HAARCASCADE:
            classifer_path = "resources/haarcascade_frontalface_default.xml"
            image = haarcascade_crop_image(image, classifer_path)

        # resize
        image = cv2.resize(image, (self.resize_res,self.resize_res), interpolation=cv2.INTER_AREA) # reduced alising effects
        
        return image

def mediapipe_crop_image(image, detection_confidence, model_type):
    face_detector = FaceDetection(detection_confidence, model_type)
    results = face_detector.process(image)
    faces = results.detections

    if faces is None:
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

def normalise(image, mean, std):
    '''does min-max then z-score normalisation'''
    return (image/255-mean)/std

def unnormalise(image, mean, std):
    return (image*std + mean)*255

def to_tensor(image, kwargs=None):
    if kwargs:
        return torch.tensor(image, device=device, **kwargs)
    
    return torch.tensor(image, device=device)