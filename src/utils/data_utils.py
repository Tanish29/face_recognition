from enum import Enum
from torch.utils.data import Dataset, random_split, DataLoader
from .datasets import *
import plotly.express as px
from tqdm import tqdm
from . import processors as proc
from albumentations import Compose
from typing import Optional

class DATASET_NAMES(Enum):
    CELEBA = 0

def get_dataset(dataset_name:DATASET_NAMES, 
                image_dir:str, 
                annotation_file:str,
                reduce_size_to:float,
                train_test_split:float,
                train_val_split:float,
                preprocessor:callable,
                transform:Optional[Compose]
                ) -> Dataset:
    """
    Returns the train, validation and test split from the given dataset

    Args:
        dataset_name: name of the dataset to load, used to call relevant function.
        image_dir: path to directory containing images 
        annotation_file: path to the annotation/label file 
        reduce_size_to: proportion of the whole dataset to keep (use arg if dataset is very large)
        train_test_split: proportion of whole dataset to use for training (e.g., 0.8 means 80% of dataset for training data)
        train_val_split: proportion of training data to use for validation
    """
    if dataset_name not in DATASET_NAMES:
        available_names = [enum.name for enum in DATASET_NAMES]
        raise ValueError(f"Provided dataset name is invalid, choose from these set of enums: {available_names}")

    # get train, val, test sizes
    train_prop = train_val_split * train_test_split
    val_prop = (1 - train_val_split) * train_test_split
    test_prop = 1 - train_test_split

    if dataset_name == DATASET_NAMES.CELEBA: # celeba dataset 
        dataset = celeba(image_dir, annotation_file, reduce_size_to, preprocessor)

    train_df, val_df, test_df = random_split(dataset, [train_prop, val_prop, test_prop])
    train_df = datasetWithTransform(train_df, transform)

    return train_df, val_df, test_df

def get_dataloaders(batch_size,*datasets):
    loaders = []
    for dataset in datasets:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)
    
    return loaders

def view_dataset(dataset:Dataset,
                 num_show:int, 
                 shuffle:bool,
                 df_name:str):
    num_images = len(dataset)
    if num_show==-1: num_show = num_images

    if shuffle:
        indices = np.random.randint(0, num_images, num_show)
    else:
        indices = range(num_show)

    num_cols = 5
    face_ids = []
    print(f"Plotting Images")
    for counter, idx in enumerate(tqdm(indices, total=len(indices))):
        image_id = dataset[idx]
        image_np = image_id[0].permute(1,2,0).cpu().numpy()
        image_np = np.expand_dims(image_np, axis=0)
        # face_ids.append(image_id[1])

        if counter == 0:
            images_np = image_np
        else:
            images_np = np.concatenate([image_np, images_np], axis=0)
    
    fig = px.imshow(images_np, 
                    animation_frame=0, 
                    title=f"Viewing {num_show} images from {df_name} dataset")
    fig.update_layout(xaxis_visible=False,
                      yaxis_visible=False)
    # for i in range(10): fig.layout.annotations[i]["text"] = f"id: {face_ids[i]}"

    fig.show()