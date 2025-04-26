import yaml

from face_recognition import DETECTOR_NAMES, PreProcessor
from face_recognition import DATASET_NAMES, get_dataset
from face_recognition import view_dataset, summarise_dataset

""" GLOBAL VARIABLES """
TRAIN_TEST_SPLIT = 0.8
TRAIN_VAL_SPLIT = 0.8
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

""" GET DATASET """
dataset_name = DATASET_NAMES.CELEBA
args = {
    "dataset_name": dataset_name,
    "image_dir": image_dir,
    "annotation_file": annotation_file,
    "reduce_size_to": REDUCE_SIZE_TO,
    "preprocessor": preprocessor,
}
df = get_dataset(**args)

""" SUMMARISE DATASET """
summarise_dataset(df)

""" VIEW DATASET """
# view_dataset(df, num_show=10, shuffle=False, df_type="train")
# view_dataset(df, num_show=10, shuffle=False, df_type="val")
