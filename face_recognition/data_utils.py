from enum import Enum
from torch.utils.data import Dataset, random_split, DataLoader
from .datasets import *
from .transforms import augment_image
import plotly.express as px
from tqdm import tqdm
from typing import Optional, Literal, List, Tuple
import os.path as osp
import pandas as pd


class DATASET_NAMES(Enum):
    CELEBA = 0


def get_image_paths_labels(
    image_dir: str, annotation_file: str
) -> Tuple[List[str], List[int]]:
    labels = pd.read_csv(
        annotation_file, sep=" ", header=None, names=["filename", "label"], dtype=str
    )
    img_paths = image_dir + osp.sep + labels.filename
    labels = labels.label.astype(np.int32)
    assert (
        img_paths.size == labels.size
    ), "Number of image paths and labels must be the same"
    valids = img_paths.apply(lambda x: osp.exists(x) and x.endswith((".png", ".jpg")))

    return img_paths[valids].to_list(), labels[valids].to_list()


def load_dataset(
    dataset_name: DATASET_NAMES,
    img_paths: list[str],
    img_labels: list[int],
    preprocessor: callable,
) -> Dataset:
    """
    Returns requested dataset

    Args:
        dataset_name: name of the dataset to load, used to call relevant function.
        img_paths: list of image file paths.
        img_labels: list of corresponding image labels.
        preprocessor: function to preprocess an image
    """
    if dataset_name not in DATASET_NAMES:
        available_names = [enum.name for enum in DATASET_NAMES]
        raise ValueError(
            f"Provided dataset name is invalid, choose from these set of enums: {available_names}"
        )

    if dataset_name == DATASET_NAMES.CELEBA:  # celeba dataset
        df = CelebA(img_paths, img_labels, preprocessor)

    return df


def split_dataset(dataset, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    """
    Splits the dataset into train, validation, and test sets.
    Args:
        dataset: The dataset to split
        train_prop: Proportion of dataset to use for training
        val_prop: Proportion of dataset to use for validation
        test_prop: Proportion of dataset to use for testing
    """
    return random_split(dataset, [train_prop, val_prop, test_prop])


def get_dataloaders(batch_size, *datasets):
    loaders = []
    for dataset in datasets:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)

    return loaders


def summarise_dataset(labels: List[int]):
    px.histogram(
        {"ID": labels},
        x="ID",
        nbins=len(labels),
        title="Distribution of Unique People in Dataset",
    ).show()


def view_dataset(
    dataset: Dataset,
    num_show: int,
    shuffle: bool,
    df_type: Literal["train", "test", "val", None],
):
    num_images = len(dataset)
    if shuffle:
        indices = np.random.randint(0, num_images, num_show)
    else:
        if num_show > num_images:
            raise ValueError(
                f"num_show ({num_show}) cannot be greater than the dataset size ({num_images})"
            )
        indices = range(num_show)

    images = np.empty((0, 3, 512, 512, 3), dtype=np.uint8)
    labels = np.zeros((num_show, 3), dtype=np.int64)
    for idx in tqdm(indices, total=num_show, desc=f"Plotting {df_type} dataset"):
        (anchor, label), (positive, plabel), (negative, nlabel) = dataset.get_item(
            idx, return_labels=True
        )
        # load images
        anchor = np.expand_dims(anchor, axis=0)
        positive = np.expand_dims(positive, axis=0)
        negative = np.expand_dims(negative, axis=0)
        image = np.concatenate((anchor, positive, negative), axis=0)
        image = np.expand_dims(image, axis=0)
        images = np.concatenate([image, images], axis=0)
        # labels
        labels[idx, :] = [label, plabel, nlabel]

    fig = px.imshow(
        images,
        animation_frame=0,
        facet_col=1,
        title=f"Viewing {num_show} images from {df_type} dataset",
    )
    fig.update_layout(xaxis_visible=False, yaxis_visible=False)
    fig.layout.annotations[0]["text"] = "Anchor"
    fig.layout.annotations[1]["text"] = "Positive"
    fig.layout.annotations[2]["text"] = "Negative"
    fig.show()

    print(f"Labels for {df_type} dataset: {labels}")
