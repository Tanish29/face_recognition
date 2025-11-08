from face_recognition import (
    DATASET_NAMES,
    load_dataset,
    get_image_paths_labels,
    get_dataloaders,
    split_dataset,
)
from face_recognition import DETECTOR_NAMES, PreProcessor
from face_recognition import view_dataset, summarise_dataset
from face_recognition.nets import SimpleNet
from face_recognition import Trainer
import torch
from torch import optim
from torch import nn

""" Configuration """
DETECTOR = DETECTOR_NAMES.MEDIAPIPE
RESIZE_RES = 512
BATCH_SIZE = 2
IMAGE_DIR = "dataset/celeba/img_celeba"
LABEL_FILE = "dataset/celeba/annotations/identity_CelebA.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1

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
view_dataset(train_df, num_show=10, shuffle=False, df_type="train")
view_dataset(val_df, num_show=10, shuffle=False, df_type="val")

""" Get Dataloaders """
train_dl, val_dl, test_dl = get_dataloaders(BATCH_SIZE, train_df, val_df, test_df)

""" Prepare For Training """
net = SimpleNet().to(DEVICE)
dummy_params = nn.Parameter(torch.zeros(1))  # to avoid error in optimisers
optimiser = optim.SGD(net.parameters(), lr=1e-2)
loss_fn = nn.TripletMarginLoss()

""" Train """
trainer = Trainer()
trainer.train(
    EPOCHS,
    DEVICE,
    net,
    train_dl,
    val_dl,
    optimiser,
    loss_fn,
)
