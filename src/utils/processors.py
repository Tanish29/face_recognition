import numpy as np
import cv2

class preprocessor():
    def __init__(self, classifer_path):
        self.classifer_path = classifer_path
    def __call__(self, image:np.ndarray):
        return self.process(image)
    def process(self, image):
        # crop 
        image = haarcascade_crop_image(image, self.classifer_path)

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
        image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 3)
    
    return image