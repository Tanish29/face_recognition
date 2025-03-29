from torch.utils.data import Dataset
from torchvision.io import read_image

import os
import numpy as np
import cv2
        
class celeba(Dataset):
    '''CelebA dataset'''
    def __init__(self, 
                 image_dir:str, 
                 annotation_file:str, 
                 reduce_size_to:int, 
                 preprocessor:callable, 
                 transform=None) -> None:
        self.image_dir = image_dir
        self.image_names = os.listdir(image_dir)
        self.image_names = np.random.choice(self.image_names, size=int(reduce_size_to*len(self.image_names)), replace=False)
        self.image_labels = np.loadtxt(annotation_file, delimiter=" ", dtype=str)
        self.preprocessor = preprocessor
        self.transform = transform
    
    def __len__(self):
        '''
        Returns the number of images 
        '''
        return len(self.image_names)
    
    def __getitem__(self, index):
        '''
        Retrieves the image and annotation at given index
        '''
        image_name = self.image_names[index]
        full_path = os.path.join(self.image_dir, image_name)

        image = cv2.imread(full_path, cv2.IMREAD_COLOR_RGB) # read image as tensor
        label = int(self.image_labels[index,1]) # get person's identity (images and annotations are ordered)

        image = self.preprocessor(image)
        if self.transform:
            image = self.transform(image)

        return image, label