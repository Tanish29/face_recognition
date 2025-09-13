import yaml
import torch


def to_tensor(image, device, **kwargs):
    return torch.tensor(image, device=device, **kwargs)
