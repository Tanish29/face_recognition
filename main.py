import yaml

from src.utils import data_utils as du
from src.utils import processors as proc

''' LOAD CONFIGURATIONS '''
with open("configs/config.yml", "r") as file: # load config file
    config = yaml.safe_load(file)  

# create variables
image_dir = config['image_dir']
annotation_file = config['label_file']
reduce_size_to = config['reduce_size_to']
train_test_split = config['train_test_split']
train_val_split = config['train_val_split']

''' PREPROCESS '''
detection_model = proc.DETECTOR_NAMES.MEDIAPIPE
detection_args = proc.get_detection_model_args(detection_model)
preprocessor = proc.preprocessor(detection_model, detection_args)

''' IMPORT DATASET '''
dataset_name = du.DATASET_NAMES.CELEBA
args = {
    "dataset_name":dataset_name,
    "image_dir":image_dir,
    "annotation_file":annotation_file,
    "reduce_size_to":reduce_size_to,
    "train_test_split":train_test_split,
    "train_val_split":train_val_split,
    "preprocessor":preprocessor
}
train_df, val_df, test_df = du.get_dataset(**args)

''' VIEW DATASET '''
du.view_dataset(train_df, resize_res=512, num_show=10, shuffle=True)