import torch
import torch.nn as nn
import weight_standardization
from weight_standardization import WeightStandardizedConv2d

class Block(nn.Module):
    '''
    Changes dimensions from : (B, C_in, H, W) -> (B, C_out, H, W)
    '''
    def __init__(self, inc, outc, groups = 8, use_ws = True):
        super().__init__()
        if (use_ws) : 
            # Used by Phil Wang who replaced the standard convolutional layer by a "weight standardized" version, 
            # which works better in combination with group normalization
            self.proj = WeightStandardizedConv2d(inc, outc, 3, padding=1)
        else : 
            self.proj = nn.Conv2d(inc, outc, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, outc)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
    

class ResnetBlock(nn.Module):
    '''
        Changes dimensions from : (B, C_in, H, W) -> (B, C_out, H, W)
    '''
    def __init__(self, inc, outc, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, outc * 2)
        ) if (time_emb_dim is not None) else None

        self.block1 = Block(inc, outc, groups = groups)
        self.block2 = Block(outc, outc, groups = groups)
        self.res_conv = nn.Conv2d(inc, outc, 1) if inc != outc else nn.Identity()

    def forward(self, x, time_emb = None):
        if (self.mlp is not None) and (time_emb is not None):
            # time_emb : (B, dim) -> (B, 2*outc)
            time_emb = self.mlp(time_emb)
            # time_emb : (B, dim) -> (B, 2*outc, 1, 1)
            time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
            # scale : (B, outc, 1, 1) & shift : (B, outc, 1, 1)
            scale_shift = time_emb.chunk(2, dim = 1)
        # h : (B, outc, H, W)
        h = self.block1(x, scale_shift = scale_shift)
        # h : (B, outc, H, W)
        h = self.block2(h)
        # result : (B, outc, H, W)
        return h + self.res_conv(x)
