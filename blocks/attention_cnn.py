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
    
att = Attention(3)

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
        # According to the paper of Efficient Attention, we need to apply the softmax to each row of q, that's why we use dim=-2
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
    
mod = LinearAttention(5)
