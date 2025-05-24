import yaml

from face_recognition import (
    DATASET_NAMES,
    get_dataset,
    get_image_paths_labels,
    get_dataloaders,
    split_dataset
)
from face_recognition import DETECTOR_NAMES, PreProcessor
from face_recognition import view_dataset, summarise_dataset

""" GLOBAL VARIABLES """
REDUCE_SIZE_TO = 0.1
DETECTOR = DETECTOR_NAMES.MEDIAPIPE
RESIZE_RES = 512
BATCH_SIZE = 32

""" LOAD CONFIGURATION FILE """
with open("configs/config.yml", "r") as file:  # load config file
    config = yaml.safe_load(file)

image_dir = config["image_dir"]  # non-platform agnostic
annotation_file = config["label_file"]

""" PREPROCESS """
preprocessor = PreProcessor(DETECTOR, RESIZE_RES)

""" get image paths and labels """
img_paths, img_labels = get_image_paths_labels(image_dir, annotation_file)

""" Get Dataset """
df = get_dataset(
    dataset_name=DATASET_NAMES.CELEBA,
    img_paths=img_paths,
    img_labels=img_labels,
    preprocessor=preprocessor
)

""" SUMMARISE DATASET """
# summarise_dataset(img_labels)

""" VIEW DATASET """
# view_dataset(df, num_show=10, shuffle=False, df_type="train")
# view_dataset(df, num_show=10, shuffle=False, df_type="val")

""" Split dataset """
train_df, val_df, test_df = split_dataset(df)
print("hello world")


