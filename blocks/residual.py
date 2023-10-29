import torch
import torch.nn as nn

# takes a function as an argument, and when called on a variable x, returns x+fn(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x