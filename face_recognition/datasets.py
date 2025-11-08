from torch.utils.data import Dataset
import numpy as np
from typing import Optional
from albumentations import Compose

from .converters import to_tensor
from .io import load_image
import copy


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
        if index >= self.__len__():
            raise IndexError(f"Index {index} out of bounds for triplet retrieval.")

        img_labels = np.array(self.img_labels)
        img_paths = np.array(self.img_paths)

        label = img_labels[index]
        pos_indices: np.ndarray = img_labels == label
        pos_indices[index] = False  # remove self from positives
        if not pos_indices.any():
            print(
                f"No positive samples found for index: {index}, using self as positive"
            )
            pos_indices[index] = True

        neg_indices: np.ndarray = ~pos_indices
        neg_indices[index] = False  # remove self from negatives
        if not neg_indices.any():
            print(
                f"No negative samples found for index: {index}, using self as negative"
            )
            neg_indices[index] = True

        anchor = img_paths[index]
        positives = tuple(zip(img_paths[pos_indices], img_labels[pos_indices]))
        positive, plabel = positives[np.random.choice(len(positives))]
        negatives = tuple(zip(img_paths[neg_indices], img_labels[neg_indices]))
        negative, nlabel = negatives[np.random.choice(len(negatives))]

        if return_labels:
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

    def __deepcopy__(self, memo):
        # Create a deep copy of the dataset
        return CelebA(
            copy.deepcopy(self.img_paths, memo),
            copy.deepcopy(self.img_labels, memo),
            self.preprocessor,  # non-picklable, use reference
        )


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
