import torch
import torch.nn as nn

from einops import rearrange
import math

class RandomOrLearnedSinusoidalPosEnc(nn.Module):

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert dim%2==0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        # x : (B, 1)
        x = rearrange(x, 'b -> b 1')
        # freqs : (B, half_dim)
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # embeddings : (B, dim)
        embbeddings = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        # embeddings : (B, dim+1) ??!
        embbeddings = torch.cat((x, embbeddings), dim = -1)
        return embbeddings