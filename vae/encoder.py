import torch 
from torch import nn 
import torch.nn.functional as F


class VAE_Encoder(nn.Sequential) : 
    def __init__(self) : 
        super.__init__()
        pass

import torch.nn.functional as F

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),           
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),            
            VAE_ResidualBlock(128, 256),             
            VAE_ResidualBlock(256, 256),             
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),            
            VAE_ResidualBlock(256, 512),            
            VAE_ResidualBlock(512, 512),             
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),             
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512),             
            VAE_ResidualBlock(512, 512),             
            VAE_AttentionBlock(512), 
            VAE_ResidualBlock(512, 512), 
            nn.GroupNorm(32, 512), 
            nn.SiLU(), 
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 
            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )


    def forward(self, x):
        for module in self : 
            if getattr(module, 'stride', None)==(2,2) : 
                x = F.pad(x , (1,0,1,0))
            x = module(x)
        return super(VAE_Encoder, self).forward(x)

# Example usage:
model = VAE_Encoder()
input_data = torch.randn(1, 10)  # Example input tensor with shape (batch_size, input_features)
output = model(input_data)
print(output)  # This will print the output tensor after passing through the network
