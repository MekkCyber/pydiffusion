import torch
import torch.nn as nn
from einops import reduce
from functools import partial
import torch.nn.functional as F

class WeightStandardizedConv2d(nn.Conv2d):
    # we dont need to define init here, we use Conv2d init method, and we only override the forward method
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        # weight : (outc, inc, H, W)
        weight = self.weight
        # we compute the mean over the first dimention which is outc, which means we fix o in outc and compute the mean across all other dimensions
        # a,,b, and c can be replaced by dots : "o ... -> o 1 1 1"
        # mean : (outc, 1, 1, 1)
        mean = reduce(weight, "o a b c -> o 1 1 1", "mean")
        # why using partial and not only torch.var directly like var = reduce(weight, "o a b c -> o 1 1 1", torch.var)
        var = reduce(weight, "o a b c -> o 1 1 1", partial(torch.var, unbiased=False))
        # normalized_weight : (outc, inc, H, W), mean and var are broadcasted to match weight size
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        # functional.conv2d is a function while nn.Conv2d is a module
        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    
conv = WeightStandardizedConv2d(2, 3, 5)
conv(torch.rand(1,2,10,10))
