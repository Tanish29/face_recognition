import yaml
import torch 

with open("configs/config.yml", "r") as file: # load config file
    config = yaml.safe_load(file)  
    DEVICE = config['device']

def to_tensor(image, **kwargs):
    return torch.tensor(image, device=DEVICE, **kwargs)