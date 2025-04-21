import albumentations as abm
import cv2
import torch
from .converters import to_tensor

transform = abm.Compose([
        abm.RandomBrightnessContrast(p=0.5), # brigthness/contrast
        abm.PixelDropout(dropout_prob=0.1, drop_value=0, p=0.5), # pixel dropout
        abm.CoarseDropout(num_holes_range=(1,5),fill="random_uniform",p=0.5), # coarse dropout
        abm.MotionBlur(p=0.5), # blur
        abm.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT,p=1.0), # shift-scale-rotate
    ])

def get_augmented_wrapper(f):
    def wrapper(*args,**kwargs):
        out = f(*args)
        image = out[0].cpu().numpy()
        image = transform(image=image)["image"]
        image = to_tensor(image)
        return *(image,out[1]),

    return wrapper
