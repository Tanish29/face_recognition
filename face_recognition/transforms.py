import albumentations as abm
import cv2
import numpy as np
from .converters import to_tensor

transform = abm.Compose(
    [
        abm.RandomBrightnessContrast(p=0.5),  # brigthness/contrast
        abm.PixelDropout(dropout_prob=0.1, drop_value=0, p=0.5),  # pixel dropout
        abm.CoarseDropout(
            num_holes_range=(1, 5), fill="random_uniform", p=0.5
        ),  # coarse dropout
        abm.MotionBlur(p=0.5),  # blur
        abm.ShiftScaleRotate(
            border_mode=cv2.BORDER_CONSTANT, p=1.0
        ),  # shift-scale-rotate
    ]
)


def augment_image(image: np.array):
    return transform(image=image)["image"]
