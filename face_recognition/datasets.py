from torch.utils.data import Dataset
import os
import numpy as np
from typing import Optional
from albumentations import Compose

from .converters import to_tensor
from .io import load_image

class CelebA(Dataset):
    """CelebA dataset"""

    def __init__(
        self,
        img_paths: list[str],
        img_labels: list[int],
        preprocessor: callable,
    ) -> None:
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.preprocessor = preprocessor
    
    def get_triplet(self, index: int) -> tuple[str]:
        # max number of triplets for random generation is len(self.img_paths)
        if index >= len(self.img_paths):
            raise IndexError(f"Index {index} out of bounds for triplet retrieval.")
        
        img_labels = np.array(self.img_labels)
        img_paths = np.array(self.img_paths)
        indices = img_labels == self.img_labels[index]
        # remove self from positives
        indices[index] = False
        if not np.any(indices):
            raise ValueError(f"No positive samples found for index: {index}")
        positives = img_paths[indices]
        # add self back 
        indices[index] = True
        if not np.any(~indices):
            raise ValueError(f"No negative samples found for index: {index}")
        negatives = img_paths[~indices]
        
        return img_paths[index], np.random.choice(positives), np.random.choice(negatives)

    def __len__(self):
        """
        Returns the number of images
        """
        return len(self.img_paths)

    def __getitem__(self, index: int):
        """
        Retrieves the image and label at given index
        """
        anchor, positive, negative = self.get_triplet(index)
        # load images
        anchor: np.ndarray = load_image(anchor)
        positive: np.ndarray = load_image(positive)
        negative: np.ndarray = load_image(negative)

        # preprocess
        anchor = self.preprocessor(anchor)
        positive = self.preprocessor(positive)
        negative = self.preprocessor(negative)

        # anchor = to_tensor(anchor).permute(2, 0, 1)
        # positive = to_tensor(positive).permute(2, 0, 1)
        # negative = to_tensor(negative).permute(2, 0, 1)

        return anchor, positive, negative


class datasetWithTransform(Dataset):
    def __init__(self, dataset: Dataset, transform: Optional[Compose]):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        if self.transform:
            image = image.permute(1, 2, 0).cpu().numpy()
            image = self.transform(image=image)["image"]
            image = to_tensor(image).permute(2, 0, 1)

        return image, label
