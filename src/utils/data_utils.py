from enum import Enum
from torch.utils.data import Dataset, random_split
from .datasets import *

class DATASET_NAMES(Enum):
    CELEBA = 0

def get_dataset(dataset_name: DATASET_NAMES, 
                image_dir: str, 
                annotation_file: str,
                reduce_size_to: float = 1.0,
                train_test_split: float = 0.8,
                train_val_split: float = 0.8,
                ) -> Dataset:
    """
    Returns the train, validation and test split from the given dataset

    Args:
        dataset_name: name of the dataset to load, used to call relevant function.
        image_dir: path to directory containing images 
        annotation_file: path to the annotation/label file 
        reduce_size_to: proportional of the whole dataset to keep (use arg if dataset is very large)
        train_test_split: proportional of whole dataset to use for training (e.g., 0.8 means 80% of dataset for training data)
        train_val_split: proportional of training data to use for validation
    """
    if dataset_name not in DATASET_NAMES:
        available_names = [enum.name for enum in DATASET_NAMES]
        raise ValueError(f"Provided dataset name is invalid, choose from these set of enums: {available_names}")
    

    # get train, val, test sizes
    train_prop = train_val_split * train_test_split
    val_prop = (1 - train_val_split) * train_test_split
    test_prop = 1 - train_test_split


    if dataset_name == DATASET_NAMES.CELEBA: # celeba dataset 
        dataset = celeba(image_dir, annotation_file, reduce_size_to)
        train_df, val_df, test_df = random_split(dataset, [train_prop, val_prop, test_prop])

    return train_df, val_df, test_df
