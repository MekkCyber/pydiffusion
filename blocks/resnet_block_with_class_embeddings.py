import torch
import torch.nn as nn
from blocks.resnet_block import Block
from einops import rearrange

class ResnetBlockWithClassEmbeddings(nn.Module):
    def __init__(self, inc, outc, *, time_emb_dim = None, classes_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), outc * 2)
        ) if time_emb_dim is not None or classes_emb_dim is not None else None

        self.block1 = Block(inc, outc, groups = groups, use_ws=False)
        self.block2 = Block(outc, outc, groups = groups, use_ws=False)
        self.res_conv = nn.Conv2d(inc, outc, 1) if inc != outc else nn.Identity()

    def forward(self, x, time_emb = None, class_emb = None):

        scale_shift = None
        if self.mlp is not None and (time_emb is not None or class_emb is not None):
            cond_emb = (time_emb, class_emb) if (time_emb is not None and class_emb is not None) else (class_emb if class_emb is not None else time_emb)
            cond_emb = torch.cat(cond_emb, dim = -1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1')
            scale_shift = cond_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)
        
        h = self.block2(h)

        return h + self.res_conv(x)

model = ResnetBlockWithClassEmbeddings(10,16,time_emb_dim=20, classes_emb_dim=16)
model(torch.rand(1,10,100,100))
