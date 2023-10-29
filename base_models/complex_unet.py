import torch
import torch.nn as nn
from functools import partial
import sys
sys.path.append('D:\\machine_learning\\projects\\stable_diffusion\\blocks')

from blocks.resnet_block import ResnetBlock
from blocks.random_sinusoidal_positional_encoding import RandomOrLearnedSinusoidalPosEnc
from blocks.sinusoidal_positional_encoding import SinusoidalPositionalEncoding
from blocks.attention_cnn import Attention, LinearAttention
from blocks.up_down_sample import UpSample, DownSample

from utils.utils import cast_tuple

class Unet(nn.Module):
    def __init__(self, dim, init_dim = None, out_dim = None, dim_mults = (1, 2, 4, 8), channels = 3, self_condition = False, resnet_block_groups = 8,
        learned_variance = False, learned_sinusoidal = False, random_sinusoidal = False, learned_sinusoidal_dim = 16, sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32, attn_heads = 4, full_attn = None, flash_attn = False):
        super().__init__()
        
        #################################Initialization######################################
        # channels = 3 for most images which is the default value
        self.channels = channels
        # The paper about self-conditioning : https://arxiv.org/pdf/2208.04202.pdf
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = init_dim if init_dim is not None else dim
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)
        
        # We increase the number of channels by double every time [init_dim, dim, 2dim, 4dim, 8dim]
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # we have in_out = [(init_dim,dim), (dim, 2*dim), (2*dim, 4*dim), (4*dim, 8*dim)] in the default case
        # its used to generate the conv layers below
        in_out = list(zip(dims[:-1], dims[1:]))
        
        #################################TIME EMBEDDING######################################
        time_dim = dim * 4
        self.random_or_learned_sinusoidal = learned_sinusoidal or random_sinusoidal
        # We check to see if we use random sinusoidal positional encoding, if so, we set random_sinusoidal to True, if we want the encoding to be random
        # which it won't be learned, or False (which is the value by default) to make it learnable
        if self.random_or_learned_sinusoidal:
            # sinus_pos_emb : (B, learned_sinusoidal_dim+1)
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEnc(learned_sinusoidal_dim, random_sinusoidal)
            encoding_dim = learned_sinusoidal_dim + 1
        else:
            # sinus_pos_emb : (B, dim)
            sinu_pos_emb = SinusoidalPositionalEncoding(dim, theta = sinusoidal_pos_emb_theta)
            encoding_dim = dim
        # The output of this layer will be of size : (B, time_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(encoding_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        #################################ATTENTION######################################
        # if we don't want the model to use full attention (full attention is the known form of attention with the scaled dot product softmax(Q@(K.T))@V)
        # we set full_attn to false, this way we will use LinearAttention (faster) except for the last layer in downsampling and upsampling
        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)
        num_stages = len(dim_mults)
        # These calls to cast_tuple will cast the variables to tuples with length num_stages which is the length of dim_mults = (1,2,4,8)
        # instead of passing single values to the UNet, we can pass tuples from the start with the parameters we want to set
        # for example instead of having attn_heads = 4, we can set different values of attention heads : attn_heads = (4,6,8,4)
        # However the length should match the length of dim_mults which the number of stages or levels in the UNet
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)
        
        #################################LAYERS######################################
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        # default case : in_out = [(init_dim,dim), (dim, 2*dim), (2*dim, 4*dim), (4*dim, 8*dim)]
        for ind, ((dim_in, dim_out), full_attn_, num_heads, dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)
            # we use partial here, becuse flash is only present in Attention, and we want attn layer to be agnostic to which layer of attention we use
            attn = partial(Attention, flash=flash_attn) if full_attn_ else LinearAttention
            
            # First iteration output : (B, dim, H//2, W//2)
            # Second iteration output : (B, 2*dim, H//4, W//4)
            # Third iteration output : (B, 4*dim, H//8, W//8)
            # Fourth iteration output : (B, 8*dim, H//8, W//8) the H, and W are not reduced because is_last is True in this case
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, groups=resnet_block_groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, groups=resnet_block_groups),
                attn(dim_in, dim_head = dim_head, heads = num_heads),
                DownSample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))
        # mid_dim is the bottleneck dimension, in the default case : 8*dim
        mid_dim = dims[-1]
        # output : (B, 8*dim, H//8, W//8)
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, groups=resnet_block_groups)
        # output : (B, 8*dim, H//8, W//8) attention doesnt change the input shape
        self.mid_attn = Attention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1], flash=flash_attn)
        # output : (B, 8*dim, H//8, W//8)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, groups=resnet_block_groups)
        # Here we try to do the reverse loop 
        for ind, ((dim_in, dim_out), full_attn_, num_heads, dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)
            attn = partial(Attention, flash=flash_attn) if full_attn_ else LinearAttention
            # First iteration : in_out[-1] = (dim_in, dim_out) = [4*dim, 8*dim]
                # First Resnet : (B, 8*dim + 4*dim, H//8, W//8) -> (B, 8*dim, H//8, W//8) (in the forward you will see why we have 8*dim + 4*dim)
                # Second Resnet : (B, 8*dim + 4*dim, H//8, W//8) -> (B, 8*dim, H//8, W//8)
                # Attention : Doesnt change the size
                # Upsample : (B, 8*dim, H//8, W//8) -> (B, 8*dim, H//4, W//4)
            # Second iteration : in_out[-2] = (dim_in, dim_out) = [2*dim, 4*dim]
                # First Resnet : (B, 4*dim + 2*dim, H//4, W//4) -> (B, 4*dim, H//4, W//4)
                # Second Resnet : (B, 4*dim + 2*dim, H//4, W//4) -> (B, 4*dim, H//4, W//4)
                # Attention : Doesnt change the size
                # Upsample : (B, 4*dim, H//4, W//4) -> (B, 2*dim, H//2, W//2)
            # Second iteration : in_out[-3] = (dim_in, dim_out) = [dim, 2*dim]
                # First Resnet : (B, 2*dim + dim, H//2, W//2) -> (B, 2*dim, H//2, W//2)
                # Second Resnet : (B, 2*dim + dim, H//2, W//2) -> (B, 2*dim, H//2, W//2)
                # Attention : Doesnt change the size
                # Upsample : (B, 2*dim, H//2, W//2) -> (B, dim, H, W)
            # Fourth iteration : in_out[-4] = (dim_in, dim_out) = [init_dim, dim]
                # First Resnet : (B, dim + init_dim, H, W) -> (B, dim, H, W)
                # Second Resnet : (B, dim + init_dim, H, W) -> (B, dim, H, W)
                # Attention : Doesnt change the size
                # Upsample : (B, dim, H, W) -> (B, init_dim, H, W) H and W don't change because this is the last iteration
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim, groups=resnet_block_groups),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim, groups=resnet_block_groups),
                attn(dim_out, dim_head = dim_head, heads = num_heads),
                UpSample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
        ]))
        # learned_variance ? To verify later what it means
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = out_dim if out_dim is not None else default_out_dim
        # We use dim*2 because we will concatenate x with its residual in the forward function
        self.final_res_block = ResnetBlock(init_dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        # we check if the height and width are divisible by 2**(num_stages-1)
        assert all([d%self.downsample_factor==0 for d in x.shape[-2:]]), f'input dimensions {x.shape[-2:]} must be divisible by {self.downsample_factor}, given the unet'
        # if self_condition is set, x : (B, 2*channels, H, W)
        if self.self_condition:
            x_self_cond = x_self_cond if x_self_cond is not None else torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim = 1)
        # x, r : (B, init_dim, H, W)
        x = self.init_conv(x)
        r = x.clone()
        # time : (B,) | t : (B, dim)
        t = self.time_mlp(time)

        h = []
        # At the end of this loop h will contain 2*(len(self.downs)) values, we will focus on the channel dimension of these values
        # h ~ [init_dim, init_dim, dim, dim, 2*dim, 2*dim, 4*dim, 4*dim]
        # in the end of the loop : x : (B, 8*dim, H//8, W//8)
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)
        # the dimensions of x won't change in this block : x : (B, 8*dim, H//8, W//8)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)
        # First iteration : 
            # we pop the last value from h : (B, 4*dim, H//8, W//8), concatenate it to x -> x : (B, 8*dim + 4*dim, H//8, W//8) 
            # you can see that the dimension matches what we saw in the self.ups loop
            # First Resnet Block : x : (B, 8*dim + 4*dim, H//8, W//8) -> (B, 8*dim, H//8, W//8) 
            # we pop again the last value from h : (B, 4*dim, H//8, W//8) it has the same dimensions as before, we concat is with x -> x : (B, 8*dim + 4*dim, H//8, W//8)
            # Second Resnet Block : x : (B, 8*dim + 4*dim, H//8, W//8) -> (B, 8*dim, H//8, W//8)
            # Attn : x : (B, 8*dim, H//8, W//8)
            # Upsample : x : (B, 4*dim, H//4, W//4)
            # You can see now why the values in the self.ups ResnetBlocks make sense
        # The same thing repeats for all iterations, in the end of the loop we have : 
        # x : (B, init_dim, H, W)
        for block1, block2, attn, upsample in self.ups:

            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)
        # x : (B, 2*init_dim, H, W)
        x = torch.cat((x, r), dim = 1)
        # x : (B, dim, H, W)
        x = self.final_res_block(x, t)
        # output : (B, out_dim, H, W)
        return self.final_conv(x)
    

model = Unet(dim=16, init_dim=48, out_dim=7)

model(torch.rand(1,3,32,32),torch.rand(1))

print(model(torch.rand(1,3,32,32),torch.rand(1)).shape)