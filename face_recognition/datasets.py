from torch.utils.data import Dataset
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

    def get_triplet(self, index: int, return_labels=False) -> tuple[str]:
        # max number of triplets for random generation is len(self.img_paths)
        if index >= len(self.img_paths):
            raise IndexError(f"Index {index} out of bounds for triplet retrieval.")

        img_labels = np.array(self.img_labels)
        img_paths = np.array(self.img_paths)

        label = img_labels[index]
        pos_indices = img_labels == label  # remove self from positives
        pos_indices[index] = False
        if not np.any(pos_indices):
            raise ValueError(f"No positive samples found for index: {index}")

        neg_indices = ~pos_indices
        neg_indices[index] = False  # add self back
        if not np.any(neg_indices):
            raise ValueError(f"No negative samples found for index: {index}")

        anchor = img_paths[index]

        positives = img_paths[pos_indices]
        pos_idx = np.random.randint(0, len(positives))
        positive = positives[pos_idx]

        negatives = img_paths[neg_indices]
        neg_idx = np.random.randint(0, len(negatives))
        negative = negatives[neg_idx]

        if return_labels:
            plabel = img_labels[pos_indices][pos_idx]
            nlabel = img_labels[neg_indices][neg_idx]
            return (anchor, label), (positive, plabel), (negative, nlabel)
        return anchor, positive, negative

    def get_item(self, index: int, return_labels=False) -> tuple[np.ndarray, str]:
        if return_labels:
            (anchor, label), (positive, plabel), (negative, nlabel) = self.get_triplet(
                index, return_labels=return_labels
            )
        else:
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

        if return_labels:
            return (anchor, label), (positive, plabel), (negative, nlabel)
        return anchor, positive, negative

    def __len__(self):
        """
        Returns the number of images.
        """
        return len(self.img_paths)

    def __getitem__(self, index: int):
        """
        Retrieves the anchor, positive and negative images for given index.
        """
        return self.get_item(index)


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
