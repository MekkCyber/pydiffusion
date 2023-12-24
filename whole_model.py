import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from torch import einsum
from functools import wraps
from packaging import version
from collections import namedtuple
from torch.cuda.amp import autocast
import math
from einops import reduce, rearrange, repeat
from collections import namedtuple
from tqdm.auto import tqdm
import random
from functools import partial
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import Adam
import math
from torchvision import utils
from ema_pytorch import EMA
from pathlib import Path
from tqdm.auto import tqdm

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1 ,t) # out = a[t] in the case of a 1D tensor a
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def identity(t, *args, **kwargs):
    return t

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        # Tensor.uniform_ : Fills tensor with numbers sampled from the continuous uniform distribution:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob
    
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def cycle(dl):
    while True:
        for data in dl:
            yield data

def function_convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


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

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class Attend(nn.Module):
    def __init__(self, dropout = 0., flash = False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu
        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        # no idea why ?!!
        if device_properties.major == 8 and device_properties.minor == 0:
            print('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):        
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        # Check if there is a compatible device for flash attention
        config = self.cuda_config if q.is_cuda else self.cpu_config
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v):
        if self.flash:
            return self.flash_attn(q, k, v)
        # q : (B, heads, H*W, dim_head)
        scale = (q.shape[-1]) ** -0.5
        # q : (B, heads, H*W, dim_head) | k : (B, heads, H*W+num_mem_kv, dim_head)
        # similarity : (B, heads, H*W, H*W+num_mem_kv)
        similarity = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale
        # attn : (B, heads, H*W, H*W+num_mem_kv)
        attn = similarity.softmax(dim = -1)
        attn = self.attn_dropout(attn)
        # out : (B, heads, H*W, dim_head)
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out


class Attention(nn.Module):
    '''
        Doesnt change the shape of input
    '''
    def __init__(self, dim, heads = 4, dim_head = 32, num_mem_kv = 4, flash = False):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        # x : (B, dim, H, W)
        x = self.norm(x)
        # (q : (B, hidden_dim, H, W) | k : (B, hidden_dim, H, W) | v : (B, hidden_dim, H, W)) 
        qkv = self.qkv(x).chunk(3, dim = 1)
        # we set h to be the number of heads, and we split the channels dimension to (h c), which means :
        # q, k, v : (B, heads, H*W, dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)
        # mk, mv : (B, heads, num_mem_kv, dim_head)
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        # k, v : (B, heads, H*W+num_mem_kv, dim_head)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))
        # out : (B, heads, H*W, dim_head)
        out = self.attend(q, k, v)
        # out : (B, hidden_dim, H, W) 
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        # out : (B, dim, H, W)
        return self.to_out(out)
    
class LinearAttention(nn.Module):
    '''
        Doesn't change the shape of input
    '''
    def __init__(self, dim, heads = 4, dim_head = 32, num_mem_kv = 4):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = RMSNorm(dim)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # x : (B, dim, H, W)
        x = self.norm(x)
        # (q : (B, hidden_dim, H, W) | k : (B, hidden_dim, H, W) | v : (B, hidden_dim, H, W)) 
        qkv = self.qkv(x).chunk(3, dim = 1)
        # q, k, v : (B, heads, head_dim, H*W)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        # mk, mv : (B, heads, head_dim, num_mem_kv)
        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        # k, v : (B, heads, head_dim, num_mem_kv+H*W)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))
        # the implementation is based on the paper https://arxiv.org/pdf/1812.01243.pdf
        # According to this paper, we need to apply the softmax to each row of q, that's why we use dim=-2
        q = q.softmax(dim = -2)
        # According to the paper we apply the softmax to each column, thus dim=-1
        k = k.softmax(dim = -1)
        q = q * self.scale
        # we compute K@V.T : (B, heads, head_dim, head_dim), this way we the complexity is reduced from O((H*W)*(H*W+num_mem_kv)) to O(head_dim*head_dim)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        # we compute context@q where q : (B, heads, head_dim, H*W) and context : (B, heads, head_dim, head_dim)
        # out : (B, heads, head_dim, H*W)
        out = torch.einsum('b h d e, b h e n -> b h d n', context, q)
        # out : (B, hidden_dim, H, W)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        # (B, dim, H, W)
        return self.to_out(out)


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

class RandomOrLearnedSinusoidalPosEnc(nn.Module):

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert dim%2==0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        device = x.device
        # x : (B, 1)
        x = rearrange(x, 'b -> b 1')
        # freqs : (B, half_dim)
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # embeddings : (B, dim)
        embbeddings = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        # embeddings : (B, dim+1) ??!
        embbeddings = torch.cat((x, embbeddings), dim = -1)
        return embbeddings.to(device)
    
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
    
class GaussianDiffusion(nn.Module):
    def __init__(self, model, *, image_size, timesteps = 1000, sampling_timesteps = None, objective = 'pred_v', beta_schedule = 'sigmoid', schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0., auto_normalize = True, offset_noise_strength = 0., min_snr_loss_weight = False, min_snr_gamma = 5):
        super().__init__()
        # why ??
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.image_size = image_size
        self.objective = objective
        # paper on distillation, appendix D to read about velocity : https://arxiv.org/pdf/2202.00512.pdf
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict velocity)'
        
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters
        self.sampling_timesteps = sampling_timesteps if sampling_timesteps is not None else timesteps

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # register_buffer is used to store models parameters that are not trained by the optimizer
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # calculations for diffusion q(x_t | x_{t-1})
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_inverse_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recip_minus_1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - claimed 0.1 was ideal
        self.offset_noise_strength = offset_noise_strength

        # loss weight : snr - signal noise ratio : https://arxiv.org/abs/2303.09556
        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_inverse_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_inverse_minus_1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_inverse_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recip_minus_1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # you can that the variance in our case here like in ddpm doesn' depend on x_start
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, device = x.shape[0], self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = unnormalize_to_zero_to_one(ret)
        return ret
    # paper : https://arxiv.org/pdf/2010.02502.pdf
    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha = extract(self.alphas_cumprod, time, shape)
            alpha_next = self.alphas_cumprod[time_next]
            alpha_next = extract(self.alphas_cumprod, torch.full((batch,),time_next,device = device, dtype = torch.long), shape)

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = unnormalize_to_zero_to_one(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps = return_all_timesteps)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, device = x1.shape[0], x1.device
        t = t if t is not None else self.num_timesteps-1

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.forward_diffusion(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def forward_diffusion(self, x_start, t, noise = None, offset_noise_strength = None):
        noise = noise if noise is not None else torch.randn_like(x_start)
        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise
        offset_noise_strength = offset_noise_strength if offset_noise_strength is not None else self.offset_noise_strength
        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None):
        b, c, h, w = x_start.shape

        noise = noise if noise is not None else torch.randn_like(x_start)

        # noise sample

        x = self.forward_diffusion(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)


class Dataset(Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png', 'tiff'], augment_horizontal_flip = False, convert_image_to = None):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.paths = self.paths[:1000]

        maybe_convert_fn = partial(function_convert_image_to, convert_image_to) if convert_image_to is not None else identity

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)
    

class Trainer(object):
    def __init__(self, diffusion_model, folder,*, train_batch_size = 16, grad_accumulation_freq = 1, augment_horizontal_flip = True, train_lr = 1e-4,
        train_num_steps = 100000, ema_update_every = 10, ema_decay = 0.995, adam_betas = (0.9, 0.99), save_and_sample_frequency = 1000, num_samples = 16,
        results_folder = '/kaggle/working/results1', amp = False, mixed_precision_type = 'fp16', split_batches = True, convert_image_to = None, calculate_fid = True,
        inception_block_idx = 2048, max_grad_norm = 1., num_fid_samples = 50000, save_best_and_latest_only = False):
        super().__init__()
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        if not convert_image_to is None:
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        assert math.sqrt(num_samples)**2==num_samples, 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_freq = save_and_sample_frequency
        self.batch_size = train_batch_size
        self.grad_accumulation_freq = grad_accumulation_freq
        assert (train_batch_size * grad_accumulation_freq) >= 16, f'Effective batch size (train_batch_size x grad_accumulation_freq) should be at least 16'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.max_grad_norm = max_grad_norm

        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)

        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True)
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        self.dl, self.opt, self.model = self.accelerator.prepare(dl, self.opt, self.model)
        self.dl = cycle(dl)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        self.step = 0
        
        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )
            # self.fid_scorer = FIDEvaluation(
            #     batch_size=self.batch_size,
            #     dl=self.dl,
            #     sampler=self.ema.ema_model,
            #     channels=self.channels,
            #     accelerator=self.accelerator,
            #     stats_dir=results_folder,
            #     device=self.device,
            #     num_fid_samples=num_fid_samples,
            #     inception_block_idx=inception_block_idx
            # )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step, 
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None,
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if self.accelerator.scaler is not None and data['scaler'] is not None:
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.grad_accumulation_freq):
                    data = next(self.dl).to(device)
                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.grad_accumulation_freq
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()
                    if self.step != 0 and self.step%self.save_and_sample_freq==0 :
                        self.ema.ema_model.eval()
                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_freq
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
                            
                        all_images = torch.cat(all_images_list, dim = 0)
                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)
                pbar.update(1)

        accelerator.print('training complete')


unet = Unet(64)
image_size = 128
diffusion_model = GaussianDiffusion(unet, image_size=image_size, timesteps=1000, sampling_timesteps=100)

trainer = Trainer(diffusion_model, '/kaggle/input/100-bird-species/valid/', train_num_steps=10000, calculate_fid=False, save_and_sample_frequency=1000)

trainer.train()
