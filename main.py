from face_recognition import (
    DATASET_NAMES,
    load_dataset,
    get_image_paths_labels,
    get_dataloaders,
    split_dataset,
)
from face_recognition import DETECTOR_NAMES, PreProcessor
from face_recognition import view_dataset, summarise_dataset

""" Configuration """
REDUCE_SIZE_TO = 0.1
DETECTOR = DETECTOR_NAMES.MEDIAPIPE
RESIZE_RES = 512
BATCH_SIZE = 32
IMAGE_DIR = ""
LABEL_FILE = ""
DEVICE = "cpu"

""" PREPROCESS """
preprocessor = PreProcessor(DETECTOR, RESIZE_RES)

""" Get image paths and labels """
img_pths, img_labs = get_image_paths_labels(IMAGE_DIR, LABEL_FILE)

""" Load Dataset """
df = load_dataset(
    dataset_name=DATASET_NAMES.CELEBA,
    img_paths=img_pths,
    img_labels=img_labs,
    preprocessor=preprocessor,
)

""" Summarize Dataset """
summarise_dataset(img_labs)

""" Split Dataset """
train_df, val_df, test_df = split_dataset(df)

""" View Dataset """
view_dataset(df, num_show=10, shuffle=False, df_type="train")
view_dataset(df, num_show=10, shuffle=False, df_type="val")

""" Get Dataloaders """
train_dl, val_dl, test_dl = get_dataloaders(BATCH_SIZE, train_df, val_df, test_df)
