import numpy as np
import cv2
from mediapipe.python.solutions.face_detection import *
from enum import Enum

normalisation_mean = (0.485, 0.456, 0.406)
normalisation_std = (0.229, 0.224, 0.225)


class DETECTOR_NAMES(Enum):
    MEDIAPIPE = "mediapipe"
    HAARCASCADE = "haarcascade"


class PreProcessor:
    def __init__(self, detection_model, resize_res, classifer_path=None):
        self.detection_model = detection_model
        self.resize_res = resize_res
        if self.detection_model == DETECTOR_NAMES.MEDIAPIPE:
            self.face_detector = FaceDetection(0.5, 1)
        elif self.detection_model == DETECTOR_NAMES.HAARCASCADE:
            self.classifier = cv2.CascadeClassifier()
            if not self.classifier.load(classifer_path):
                raise ValueError(f"Error: failed to load face classifier")

    def __call__(self, image: np.ndarray):
        return self.process(image)

    def process(self, image):
        # crop
        try:
            if self.detection_model == DETECTOR_NAMES.MEDIAPIPE:
                image_cropped = self.mediapipe_crop_image(image)
            elif self.detection_model == DETECTOR_NAMES.HAARCASCADE:
                image_cropped = self.haarcascade_crop_image(image)

            if image_cropped.size == 0:
                image_cropped = image
        except Exception as e:
            print(f"Error during cropping: {e}. Skipping step.")
            image_cropped = image

        # resize
        image_resized = cv2.resize(
            image_cropped,
            (self.resize_res, self.resize_res),
            interpolation=cv2.INTER_AREA,
        )  # reduced alising effects

        return image_resized


    def mediapipe_crop_image(self, image):
        results = self.face_detector.process(image)
        faces = results.detections

        if faces is None:
            print(f"Log: No faces detected returning unchanged image")
            return image

        h, w, _ = image.shape  # (H,W,C)
        for face in faces:
            bbox = face.location_data.relative_bounding_box
            x, y, w, h = (
                int(bbox.xmin * w),
                int(bbox.ymin * h),
                int(bbox.width * w),
                int(bbox.height * h),
            )
            # image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            image = image[y : y + h, x : x + w]

        return image


    def haarcascade_crop_image(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.equalizeHist(image_gray)

        faces = self.classifier.detectMultiScale(image_gray)
        for face in faces:
            x, y, w, h = face
            # image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            image = image[y : y + h, x : x + w]

        return image


def normalise(image, mean, std):
    """does min-max then z-score normalisation"""
    return (image / 255 - mean) / std


def unnormalise(image, mean, std):
    return (image * std + mean) * 255
