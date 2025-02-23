from torch.utils.data import Dataset
from torchvision.io import read_image

import os
import numpy as np
        
class celeba(Dataset):
    '''CelebA dataset'''
    def __init__(self, image_dir, annotation_file, transform=None, target_transform=None) -> None:
        self.image_dir = image_dir
        self.image_labels = np.loadtxt(annotation_file, delimiter=" ", dtype=str)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        '''
        Returns the number of images 
        '''
        return len(os.listdir(self.image_dir))
    
    def __getitem__(self, index):
        '''
        Retrieves the image and annotation at given index
        '''
        image_name = os.listdir(self.image_dir)[index]
        full_path = os.path.join(self.image_dir, image_name)

        image = read_image(full_path) # read image as tensor
        label = int(self.image_labels[index,1]) # get person's identity (images and annotations are ordered)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label