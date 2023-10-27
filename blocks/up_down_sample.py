import torch
import torch.nn as nn



class UpSample(nn.Module) : 
    def __init__(self, dim, dim_out=None) : 
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(dim, dim_out if dim_out is not None else dim, 3, padding = 1)

    def forward(self, x) : 
        return self.conv(self.up(x))
    

class DownSample(nn.Module) : 
    # this implementation differs from the official one
    def __init__(self, dim, dim_out=None) : 
        super().__init__()
        self.down = nn.Conv2d(dim, dim, 3, 2, 1)
        self.conv = nn.Conv2d(dim, dim_out if dim_out is not None else dim, 3,1,1)
    def forward(self, x) : 
        return self.conv(self.down(x))