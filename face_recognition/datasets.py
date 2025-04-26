from torch.utils.data import Dataset
import os
import numpy as np
from typing import Optional
from albumentations import Compose

from .converters import to_tensor
from .io import load_image


class celeba(Dataset):
    """CelebA dataset"""

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        reduce_size_to: Optional[int],
        preprocessor: callable,
    ) -> None:
        self.image_dir = image_dir
        self.image_names = os.listdir(image_dir)
        if reduce_size_to is not None:
            self.image_names = np.random.choice(
                self.image_names,
                size=int(reduce_size_to * len(self.image_names)),
                replace=False,
            )
        self.image_labels = np.loadtxt(annotation_file, delimiter=" ", dtype=str)
        self.preprocessor = preprocessor

    def __len__(self):
        """
        Returns the number of images
        """
        return len(self.image_names)

    def __getitem__(self, index: int):
        """
        Retrieves the image and annotation at given index
        """
        image_name = self.image_names[index]
        full_path = os.path.join(self.image_dir, image_name)

        image: np.ndarray = load_image(full_path)
        label = int(
            self.image_labels[index, 1]
        )  # get person's identity (images and annotations are ordered)

        # image = self.preprocessor(image)

        # image = to_tensor(image).permute(2, 0, 1)
        # label = to_tensor(label)

        return image, label


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
