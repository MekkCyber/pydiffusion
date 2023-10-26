import torch
import torch.nn as nn
import torch.nn.functional as F

class UnetBlock(nn.Module) : 
    def __init__(self, inc, outc, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, outc)
        if up:
            self.conv1 = nn.Conv2d(2*inc, outc, 3, padding=1)
            self.transform = nn.ConvTranspose2d(outc, outc, 4, 2, 1)
            nn.GroupNorm()
        else:
            self.conv1 = nn.Conv2d(inc, outc, 3, padding=1)
            self.transform = nn.Conv2d(outc, outc, 4, 2, 1)
        self.conv2 = nn.Conv2d(outc, outc, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(outc)
        self.bnorm2 = nn.BatchNorm2d(outc)
        self.relu  = nn.ReLU()

    def forward(self, x, t) :
        h = self.bnorm1(self.relu(self.conv1(x)))
        time = self.relu(self.time_mlp(t))
        time = time.unsqueeze(-1).unsqueeze(-1)
        h = h + time
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


