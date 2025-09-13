from torch import nn
import torch


class SimpleNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dummy_params = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """A simple network that returns the input image as embedding (flattened)"""
        x = x.view(x.shape[0], -1)  # flatten (B, C, H, W) to (B, C*H*W)
        x = x.float().requires_grad_(True)
        return x
