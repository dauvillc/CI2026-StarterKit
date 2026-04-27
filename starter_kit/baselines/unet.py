# COPIED FROM THE AMAZING LUCIDRAINS https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/guided_diffusion.py

import logging
from typing import Dict, Any
import math
# import copy
# from pathlib import Path
# from random import random
from functools import partial
from collections import namedtuple
# from multiprocessing import cpu_count

# Internal modules
from starter_kit.model import BaseModel
from starter_kit.layers import InputNormalisation

import torch
from torch import nn, einsum
import torch.nn.functional as F
# from torch.cuda.amp import autocast

from einops import rearrange  # , reduce, repeat

# from einops.layers.torch import Rearrange

# from tqdm.auto import tqdm


# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

_normalisation_mean = [
    294.531359,287.010605,278.507482,262.805241,227.580722,201.364517,
    209.719502,0.010667,0.006922,0.003784,0.001229,0.000088,0.000003,
    0.000003,-1.412110,-0.914917,0.431349,3.504875,11.699176,6.758849,
    -1.214763,0.167424,-0.105374,-0.172138,-0.022648,0.030789,0.281048,
    -0.094608,0.410844,2129.684371
]
_normalisation_std = [
    62.864550,61.180621,58.938862,56.016099,47.532073,32.281805,38.084321,
    0.006102,0.004648,0.003013,0.001266,0.000080,0.000001,0.000000,4.661358,
    6.159993,7.763541,9.877940,16.068963,11.681901,10.705570,4.119853,4.318767,
    4.810067,6.209760,10.585627,5.680168,2.978756,0.498762,3602.712270
]

# helpers functions

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# classifier free guidance functions

def uniform(shape, device):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, dim_aux=2, groups=8):
        super().__init__()

        self.block1 = Block(dim+dim_aux, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, aux):
        h = self.block1(torch.cat((x, aux), dim=1))

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(
            self,
            num_classes,
            in_channels,
            out_channels,
            dim=128,
            dim_mults=(1, 2, 2, 2),  # ,(1, 2, 4, 8),
            resnet_block_groups=8,
            attn_dim_head=64,
            attn_heads=4,
            use_classes=False,
            init_groups=1
    ):

        super().__init__()

        # determine dimensions

        self.use_classes = use_classes

        self.in_channels = in_channels
        input_channels = in_channels

        init_dim = dim
        dim_aux = 2
        hidden_init = init_dim // init_groups * init_groups
        self.init_conv = nn.Sequential(nn.Conv2d(input_channels, hidden_init, 7, padding=3, groups=init_groups),
                                       nn.Conv2d(hidden_init, init_dim, 7, padding=3))

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                Downsample(dim_aux, dim_aux) if not is_last else nn.Conv2d(dim_aux, dim_aux, 3, padding=1),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out),
                block_klass(dim_out + dim_in, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
            ]))

        self.out_channels = out_channels

        self.final_res_block = block_klass(dim * 2, dim)
        self.final_conv = nn.Conv2d(dim, self.out_channels, 1)

        self.normalisation = InputNormalisation(
            mean=torch.tensor(_normalisation_mean),
            std=torch.tensor(_normalisation_std)
        )

        logging.info(self)

    def forward(self, input_level, input_auxiliary):

        # We collapse all levels into the channel dimension
        flattened_input_level = input_level.reshape(
            input_level.shape[0], -1, *input_level.shape[-2:]
        )

        sliced_auxiliary = input_auxiliary[:, :2]

        norm = torch.cat([
            flattened_input_level,
            sliced_auxiliary
        ], dim=1).movedim(1, -1)

        # Apply input normalisation
        normed = self.normalisation(norm).movedim(-1, 1)

        x, aux = normed[:, :-2], normed[:, -2:]

        # print("\n Shape at the beginning:", x.shape)
        x = self.init_conv(torch.cat((x, aux), dim=1))
        # print("After init_conv:", x.shape)
        r = x.clone()

        h = []
        h_aux = []

        for block1, block2, attn, downsample, downaux in self.downs:
            x = block1(x, aux)
            # print("After block1:", x.shape)
            h.append(x)

            x = block2(x, aux)
            # print("After block2:", x.shape)
            x = attn(x)
            # print("After attn:", x.shape)
            h.append(x)
            h_aux.append(aux)

            x = downsample(x)
            aux = downaux(aux)
            # print("After downsample:", x.shape)

        x = self.mid_block1(x, aux)
        # print("After mid_block1:", x.shape)
        x = self.mid_attn(x)
        # print("After mid_attn:", x.shape)
        x = self.mid_block2(x, aux)
        # print("After mid_block2:", x.shape)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            aux = h_aux.pop()
            # print("After unet_pop:", x.shape)
            x = block1(x, aux)
            # print("After block1:", x.shape)

            x = torch.cat((x, h.pop()), dim=1)
            # print("After unet_pop:", x.shape)
            x = block2(x, aux)
            # print("After block2:", x.shape)
            x = attn(x)
            # print("After attn:", x.shape)

            x = upsample(x)
            # print("After upsample:", x.shape)

        x = torch.cat((x, r), dim=1)
        # print("After unet_pop:", x.shape)

        x = self.final_res_block(x, aux)
        # print("After final_res_block:", x.shape)
        x = self.final_conv(x)
        # print("After final_conv:", x.shape)
        return x

class UnetModel(BaseModel):
    r'''
    Model wrapper for an MLP network with standard loss outputs.

    This class delegates forward execution to a hidden MLP network and
    computes a mean absolute error loss together with auxiliary metrics.
    '''

    def estimate_loss(
            self,
            batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        r'''
        Compute the primary training loss and prediction output.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dictionary containing ``input_level``,
            ``input_auxiliary``, and ``target`` tensors.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys ``loss`` and ``prediction``.
            ``loss`` is the mean absolute error and ``prediction`` is the
            model output clamped to ``[0, 1]``.
        '''
        prediction = self.network(
            input_level=batch["input_level"],
            input_auxiliary=batch["input_auxiliary"]
        )
        prediction = prediction.clamp(0., 1.)
        loss = (prediction - batch["target"]).abs()
        loss = loss * self.lat_weights
        loss = loss.mean()
        return {"loss": loss, "prediction": prediction}

    def estimate_auxiliary_loss(
            self,
            batch: Dict[str, torch.Tensor],
            outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        r'''
        Compute auxiliary regression and classification metrics.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dictionary containing the ground-truth ``target`` tensor.
        outputs : Dict[str, Any]
            Model outputs from ``estimate_loss`` containing ``prediction``.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys ``mse`` and ``accuracy``.
            ``mse`` is the mean squared error and ``accuracy`` is the
            thresholded classification accuracy at 0.5.
        '''
        mse = (outputs["prediction"] - batch["target"]).pow(2)
        mse = (mse * self.lat_weights).mean()
        prediction_bool = (outputs["prediction"] > 0.5).float()
        target_bool = (batch["target"] > 0.5).float()
        accuracy = (prediction_bool == target_bool).float()
        accuracy = (accuracy * self.lat_weights).mean()
        return {"mse": mse, "accuracy": accuracy}