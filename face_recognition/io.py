import mmap
import numpy as np
import cv2


def load_image(path: str) -> np.ndarray:
    with open(path, "r") as f:
        file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        array = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR_RGB)
        file.close()

    return image
