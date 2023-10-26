import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from torch import einsum

from rms_norm import RMSNorm

from functools import wraps
from packaging import version
from collections import namedtuple

AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

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
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
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
        scale = (q.shape[-1]*q.shape[1]) ** -0.5
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
        # x : (B, C, H, W) where C is the hidden dimension
        x = self.norm(x)
        # (q : (B, C, H, W) | k : (B, C, H, W) | v : (B, C, H, W))
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
        # out : (B, C, H, W) where C is the hidden dimension
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        # out : (B, C, H, W) where C is the hidden dimension
        return self.to_out(out)
    
att = Attention(3)
print(att(torch.rand(1,3,10,10)))
