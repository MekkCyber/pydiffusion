import torch
import torch.nn as nn
from functools import partial
from einops import rearrange, repeat

from blocks.resnet_block_with_class_embeddings import ResnetBlockWithClassEmbeddings
from blocks.sinusoidal_positional_encoding import SinusoidalPositionalEncoding
from blocks.random_sinusoidal_positional_encoding import RandomOrLearnedSinusoidalPosEnc
from blocks.attention_cnn import Attention, LinearAttention
from blocks.up_down_sample import UpSample, DownSample
from blocks.residual import Residual
from blocks.prenorm import PreNorm
from utils.utils import prob_mask_like

class UnetClassifierFree(nn.Module):
    def __init__(self, dim, num_classes, condition_class_drop_prob = 0.5, init_dim = None, out_dim = None, dim_mults=(1, 2, 4, 8), channels = 3, resnet_block_groups = 8,
        learned_variance = False, learned_sinusoidal = False, random_sinusoidal = False, learned_sinusoidal_dim = 16, attn_dim_head = 32, attn_heads = 4
    ):
        super().__init__()
        #################################Initialization######################################
        # channels = 3 for most images which is the default value
        self.channels = channels
        input_channels = channels
        # This is the probability by which we can drop class conditionning, to perform normal diffusion
        self.condition_class_drop_prob = condition_class_drop_prob
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
            sinu_pos_emb = SinusoidalPositionalEncoding(dim)
            encoding_dim = dim
        # The output of this layer will be of size : (B, time_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(encoding_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        #################################Class EMBEDDING######################################
        # There is no positional encoding for the classes so we just use the Embedding layer
        # classes_emb : (num_classes, dim)
        self.classes_emb = nn.Embedding(num_classes, dim)
        # in classifier free guidance, we sometimes use the class embeddings, and sometimes not (according to the condition_class_drop_prob probability)
        # when we sample without the class embeddings we use null_classes_emb which is a random vector of size (dim,)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))
        classes_dim = dim * 4
        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        #################################LAYERS######################################
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlockWithClassEmbeddings(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim, groups = resnet_block_groups),
                ResnetBlockWithClassEmbeddings(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim, groups = resnet_block_groups),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                DownSample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlockWithClassEmbeddings(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim, groups = resnet_block_groups)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = ResnetBlockWithClassEmbeddings(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim, groups = resnet_block_groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlockWithClassEmbeddings(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim, groups = resnet_block_groups),
                ResnetBlockWithClassEmbeddings(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim, groups = resnet_block_groups),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                UpSample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = out_dim if out_dim is not None else default_out_dim

        self.final_res_block = ResnetBlockWithClassEmbeddings(init_dim * 2, dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
    # From the arguments passed we only keep those we need for this method, the other arguments necessary for the forward method will be stored in *args and **kwargs
    def forward_with_condition_scale(self, *args, condition_scale = 1., rescaled_phi = 0., **kwargs):
        # condition_class_drop_prob = 0 -> we use the model with class embeddings
        logits = self.forward(*args, condition_class_drop_prob = 0., **kwargs)
        # Here the condition_scale is used to define how much we keep from the logits when they contain the class embeddings (logits are the predicted noise)
        if condition_scale == 1:
            return logits
        # if the condition_scale is not set to 1, it means we will interpolate between a class free logits and a class guided logits
        # The null_logits are associated with condition_class_drop_prob = 1 -> we use the model with null class embeddings 
        null_logits = self.forward(*args, condition_class_drop_prob = 1., **kwargs)
        # The formula is present in the paper of Classifier Free Guidance : https://arxiv.org/pdf/2207.12598.pdf, Page : 5, Formula : 6, set (w + 1) to condition_scale
        scaled_logits = null_logits + (logits - null_logits) * condition_scale

        if rescaled_phi == 0.:
            return scaled_logits

        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

    def forward(self, x, time, classes, condition_class_drop_prob = None):
        b, device = x.shape[0], x.device

        condition_class_drop_prob = condition_class_drop_prob if condition_class_drop_prob is not None else self.condition_class_drop_prob
        # classes_emb : (B, classes_dim)
        classes_emb = self.classes_emb(classes)
        # if condition_class_drop_prob > 0, it means we will have some null_classes
        # prob_mask_like creates a vector of booleans, where True is present with a probability of (1-condition_class_drop_prob) which means True -> keep class
        if condition_class_drop_prob > 0:
            # keep_mask : (B,)
            keep_mask = prob_mask_like((b,), 1 - condition_class_drop_prob, device = device)
            # null_classes_emb : (B, dim)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = b)
            # classes_emb : (B, dim)
            # torch.where broadcasts rearrange(keep_mask, 'b -> b 1') to have shape (B, dim)
            # when we have True in keep_mask, its broadcasted to be (true, true,..., true), so we keep all values in the classes_emb corresponding row
            # same happens when we have false, we store in classes_emb the embeddings of null_classes_emb
            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )
        # c : (B, classes_dim)
        c = self.classes_mlp(classes_emb)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c)

        return self.final_conv(x)


model = UnetClassifierFree(8,100,init_dim=8, channels=77)
model(torch.rand(1,77,200,200), torch.rand(1),torch.tensor([4], dtype=torch.long)).shape