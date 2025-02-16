from enum import Enum
from datasets import *

class DATASET_NAMES(Enum):
    celeba = 0

def get_dataset(dataset_name: DATASET_NAMES):
    """
    Returns the train, validation and test split from the given dataset

    Args:
        dataset_name: name of the dataset to load, used to call relevant function. 
    """
    if dataset_name not in DATASET_NAMES:
        available_names = [enum.name for enum in DATASET_NAMES]
        raise ValueError(f"Provided dataset name is invalid, choose from these set of enums: {available_names}")
    
    if dataset_name == DATASET_NAMES.CELEBA: # celeba dataset 
        train_df, val_df, test_df = celeba()


    return train_df, val_df, test_df

