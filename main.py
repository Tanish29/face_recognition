import yaml

from face_recognition import DETECTOR_NAMES, get_preprocessor_args, preprocessor
from face_recognition import DATASET_NAMES, get_dataset, get_dataloaders
from face_recognition import view_dataset
from face_recognition import get_albumentation_transform

''' GLOBAL VARIABLES '''
TRAIN_TEST_SPLIT = 0.8
TRAIN_VAL_SPLIT = 0.8
REDUCE_SIZE_TO = 0.10
DETECTOR = DETECTOR_NAMES.MEDIAPIPE
RESIZE_RES = 512
BATCH_SIZE = 32

''' LOAD CONFIGURATION FILE '''
with open("configs/config.yml", "r") as file: # load config file
    config = yaml.safe_load(file)  

image_dir = config['image_dir'] # non-platform agnostic
annotation_file = config['label_file']

''' PREPROCESS '''
preprocessor_args = get_preprocessor_args(DETECTOR,RESIZE_RES)
preprocessor = preprocessor(preprocessor_args)

''' AUGMENTATIONS '''
transform = get_albumentation_transform()

''' GET DATASET '''
dataset_name = DATASET_NAMES.CELEBA
args = {
    "dataset_name":dataset_name,
    "image_dir":image_dir,
    "annotation_file":annotation_file,
    "reduce_size_to":REDUCE_SIZE_TO,
    "train_test_split":TRAIN_TEST_SPLIT,
    "train_val_split":TRAIN_VAL_SPLIT,
    "preprocessor":preprocessor,
    "transform":None
}
train_df, val_df, test_df = get_dataset(**args)

''' GET DATALOADER '''
train_dl, val_dl, test_dl = get_dataloaders(BATCH_SIZE,train_df,val_df,test_df)

''' VIEW DATASET '''
view_dataset(train_df, num_show=10, shuffle=False, df_name="Train")
view_dataset(val_df, num_show=10, shuffle=True, df_name="val")