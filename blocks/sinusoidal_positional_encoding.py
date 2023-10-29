import torch 
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module) : 
    def __init__(self, dim, theta = None) : 
        super().__init__()
        self.dim = dim
        
    def forward(self, t) : 
        device = t.device
        t = t.type(torch.float)
        half_dim = self.dim // 2
        denominator = math.log(1000)/(half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device)*(-denominator))
        # t : (b) & embeddings : (dim/2) -> embeddings : (b, dim/2)
        embeddings = t.unsqueeze(1)@embeddings.unsqueeze(0)
        # embeddings : (b, dim/2) -> embeddings : (b,dim)
        embeddings = torch.concat((embeddings.sin(), embeddings.cos()), dim=1)
        return embeddings
