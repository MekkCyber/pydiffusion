import torch
import torch.nn as nn

from blocks.rms_norm import RMSNorm

# This block normalizes the data before applying the function fn
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)