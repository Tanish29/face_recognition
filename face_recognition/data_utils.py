from enum import Enum
from torch.utils.data import Dataset, random_split, DataLoader
from .datasets import *
from .transforms import augment_image
import plotly.express as px
from tqdm import tqdm
from typing import Optional, Literal

class DATASET_NAMES(Enum):
    CELEBA = 0

def get_dataset(dataset_name:DATASET_NAMES, 
                image_dir:str, 
                annotation_file:str,
                reduce_size_to:Optional[float],
                preprocessor:callable
                ) -> Dataset:
    """
    Returns requested dataset

    Args:
        dataset_name: name of the dataset to load, used to call relevant function.
        image_dir: path to directory containing images 
        annotation_file: path to the annotation/label file 
        reduce_size_to: proportion of the whole dataset to keep (use arg if dataset is very large)
    """
    if dataset_name not in DATASET_NAMES:
        available_names = [enum.name for enum in DATASET_NAMES]
        raise ValueError(f"Provided dataset name is invalid, choose from these set of enums: {available_names}")

    # get train, val, test sizes
    # train_prop = train_val_split * train_test_split
    # val_prop = (1 - train_val_split) * train_test_split
    # test_prop = 1 - train_test_split
    
    if dataset_name == DATASET_NAMES.CELEBA: # celeba dataset 
        df = celeba(image_dir, annotation_file, reduce_size_to, preprocessor)

    # train_df, val_df, test_df = random_split(dataset, [train_prop, val_prop, test_prop])
    # train_df = datasetWithTransform(train_df, transform)

    return df

def get_dataloaders(batch_size,*datasets):
    loaders = []
    for dataset in datasets:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)
    
    return loaders

def summarise_dataset(dataset:Dataset):
    num_images = len(dataset)

    labels = list(tqdm(map(lambda x: dataset[x][1].item(), range(num_images)), 
                       total=num_images, desc=f"Retrieving labels for {num_images} images"))

    fig = px.histogram({"ID":labels}, x="ID", nbins=num_images, title="Distribution of Unique People in Dataset")
    fig.show()


def get_dataset_at_idx(dataset, idx, augment=False):
    image_id = dataset[idx]
    image = image_id[0].permute(1,2,0).cpu().numpy()
    iid = image_id[1].item()

    if augment:
        image = augment_image(image)

    return image, iid

def view_dataset(dataset:Dataset,
                 num_show:int, 
                 shuffle:bool,
                 df_type:Literal['train','test','val',None]):
    num_images = len(dataset)
    if num_show==-1: 
        num_show = num_images

    if shuffle:
        indices = np.random.randint(0, num_images, num_show)
    else:
        indices = range(num_show)

    num_cols = 5
    face_ids = []
    augment = True if df_type == "train" else False

    for counter, idx in enumerate(tqdm(indices, desc=f"Plotting")):
        image, iid = get_dataset_at_idx(dataset,idx,augment=augment)
        image = np.expand_dims(image, axis=0)
        # face_ids.append(image_id[1])

        if counter == 0:
            images = image
        else:
            images = np.concatenate([image, images], axis=0)
    
    fig = px.imshow(images, 
                    animation_frame=0, 
                    title=f"Viewing {num_show} images from {df_type} dataset")
    fig.update_layout(xaxis_visible=False,
                      yaxis_visible=False)
    # for i in range(10): fig.layout.annotations[i]["text"] = f"id: {face_ids[i]}"

    fig.show()