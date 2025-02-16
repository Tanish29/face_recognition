from torch.utils.data import Dataset
from torchvision.io import read_image
import os

class celeba(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, target_transform=None) -> None:
        self.image_dir = image_dir
        self.image_labels = annotation_file
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(os.listdir(self.image_dir))
    
    def __getitem__(self, index):
        image_name = os.listdir(self.image_dir)[index]
        full_path = os.join.path(self.image_dir, image_name)

        image = read_image(full_path) # read image as tensor
        label = self.image_labels[index] # get person's identity

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    