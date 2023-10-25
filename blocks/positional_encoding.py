import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module) : 
    def __init__(self, dim) : 
        super().__init__()
        self.dim = dim
    # converts time steps to embedded time steps
    # N -> (N, dim)
    def forward(self, time) : 
        denom = torch.tensor(10000)**((2*torch.arange(self.dim))/self.dim)
        time = time.type(torch.float)
        embeddings = time.unsqueeze(1)@denom.unsqueeze(0)
        embeddings[:,0::2] = torch.cos(embeddings[:,0::2])
        embeddings[:,1::2] = torch.sin(embeddings[:,1::2])
        return embeddings

