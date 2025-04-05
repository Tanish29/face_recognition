import yaml

from src.utils import data_utils as du
from src.utils import processors as proc

''' GLOBAL VARIABLES '''
TRAIN_TEST_SPLIT = 0.8
TRAIN_VAL_SPLIT = 0.8
REDUCE_SIZE_TO = 0.10
DETECTOR = proc.DETECTOR_NAMES.MEDIAPIPE
RESIZE_RES = 512

''' LOAD CONFIGURATION FILE '''
with open("configs/config.yml", "r") as file: # load config file
    config = yaml.safe_load(file)  

image_dir = config['image_dir'] # non-platform agnostic
annotation_file = config['label_file']

''' PREPROCESS '''
preprocessor_args = proc.get_preprocessor_args(DETECTOR,RESIZE_RES)
preprocessor = proc.preprocessor(preprocessor_args)

''' IMPORT DATASET '''
dataset_name = du.DATASET_NAMES.CELEBA
args = {
    "dataset_name":dataset_name,
    "image_dir":image_dir,
    "annotation_file":annotation_file,
    "reduce_size_to":REDUCE_SIZE_TO,
    "train_test_split":TRAIN_TEST_SPLIT,
    "train_val_split":TRAIN_VAL_SPLIT,
    "preprocessor":preprocessor
}
train_df, val_df, test_df = du.get_dataset(**args)

''' VIEW DATASET '''
du.view_dataset(train_df, num_show=10, shuffle=True, df="Train")