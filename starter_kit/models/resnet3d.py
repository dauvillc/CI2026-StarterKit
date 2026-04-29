#!/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from starter_kit.layers import InputNormalisation
from starter_kit.model import BaseModel

main_logger = logging.getLogger(__name__)

# Pre-computed mean/std from training data: 28 level features + 2 aux features.
# Same values as cnn.py.
_normalisation_mean = [
    294.531359,
    287.010605,
    278.507482,
    262.805241,
    227.580722,
    201.364517,
    209.719502,
    0.010667,
    0.006922,
    0.003784,
    0.001229,
    0.000088,
    0.000003,
    0.000003,
    -1.412110,
    -0.914917,
    0.431349,
    3.504875,
    11.699176,
    6.758849,
    -1.214763,
    0.167424,
    -0.105374,
    -0.172138,
    -0.022648,
    0.030789,
    0.281048,
    -0.094608,
    0.410844,
    2129.684371,
]
_normalisation_std = [
    62.864550,
    61.180621,
    58.938862,
    56.016099,
    47.532073,
    32.281805,
    38.084321,
    0.006102,
    0.004648,
    0.003013,
    0.001266,
    0.000080,
    0.000001,
    0.000000,
    4.661358,
    6.159993,
    7.763541,
    9.877940,
    16.068963,
    11.681901,
    10.705570,
    4.119853,
    4.318767,
    4.810067,
    6.209760,
    10.585627,
    5.680168,
    2.978756,
    0.498762,
    3602.712270,
]


# ── Weight inflation helpers ──────────────────────────────────────────────────


def _inflate_first_conv(
    w2d: torch.Tensor, new_in_channels: int, temporal_depth: int
) -> torch.Tensor:
    """
    Inflate the stem Conv2d(3, C_out, kH, kW) weight to Conv3d.

    Averages over the 3 RGB input channels, expands to new_in_channels,
    then stacks temporal_depth copies divided by temporal_depth so the
    summed response matches the 2D baseline.

    Returns shape (C_out, new_in_channels, temporal_depth, kH, kW).
    """
    w = w2d.mean(dim=1, keepdim=True)  # (C_out, 1, kH, kW)
    w = w.expand(-1, new_in_channels, -1, -1)  # (C_out, new_in_channels, kH, kW)
    w = w.unsqueeze(2).expand(-1, -1, temporal_depth, -1, -1).clone()
    return w / temporal_depth  # (C_out, new_in_channels, kT, kH, kW)


def _inflate_conv(w2d: torch.Tensor, temporal_depth: int) -> torch.Tensor:
    """
    Inflate a block Conv2d(C_in, C_out, kH, kW) weight to Conv3d.

    Stacks temporal_depth copies along the new depth axis, divided by
    temporal_depth to preserve the summed activation magnitude.

    Returns shape (C_out, C_in, temporal_depth, kH, kW).
    """
    w = w2d.unsqueeze(2).expand(-1, -1, temporal_depth, -1, -1).clone()
    return w / temporal_depth


def _inflate_shortcut(w2d: torch.Tensor) -> torch.Tensor:
    """
    Lift a 1×1 shortcut Conv2d weight to Conv3d with kT=1 (no temporal mixing).

    Returns shape (C_out, C_in, 1, 1, 1).
    """
    return w2d.unsqueeze(2)


def _copy_bn(src: nn.BatchNorm2d, dst: nn.BatchNorm3d) -> None:
    """Copy all learned and running statistics from BatchNorm2d into BatchNorm3d."""
    if src.running_mean is None or src.running_var is None:
        raise ValueError("Source BatchNorm2d must have running_mean and running_var buffers.")
    if dst.running_mean is None or dst.running_var is None:
        raise ValueError("Destination BatchNorm3d must haverunning_mean and running_var buffers.")
    dst.weight.data.copy_(src.weight.data)
    dst.bias.data.copy_(src.bias.data)
    dst.running_mean.copy_(src.running_mean)
    dst.running_var.copy_(src.running_var)
    dst.eps = src.eps
    dst.momentum = src.momentum


# ── 3D BasicBlock ─────────────────────────────────────────────────────────────


class BasicBlock3D(nn.Module):
    """3D residual block assembled from inflated 2D BasicBlock components."""

    def __init__(
        self,
        conv1: nn.Conv3d,
        bn1: nn.BatchNorm3d,
        conv2: nn.Conv3d,
        bn2: nn.BatchNorm3d,
        relu: nn.ReLU,
        downsample: Optional[nn.Sequential] = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.conv2 = conv2
        self.bn2 = bn2
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _inflate_basic_block(block: Any, temporal_depth: int) -> BasicBlock3D:
    """
    Build a BasicBlock3D from a torchvision BasicBlock with inflated weights.

    All spatial strides are forced to 1 to preserve the 64×64 grid resolution.
    """
    in_ch = block.conv1.in_channels
    out_ch = block.conv1.out_channels
    kT = temporal_depth
    pad_t = kT // 2

    conv1 = nn.Conv3d(
        in_ch, out_ch, (kT, 3, 3), stride=(1, 1, 1), padding=(pad_t, 1, 1), bias=False
    )
    conv1.weight.data.copy_(_inflate_conv(block.conv1.weight.data, kT))

    bn1 = nn.BatchNorm3d(out_ch)
    _copy_bn(block.bn1, bn1)

    conv2 = nn.Conv3d(
        out_ch, out_ch, (kT, 3, 3), stride=(1, 1, 1), padding=(pad_t, 1, 1), bias=False
    )
    conv2.weight.data.copy_(_inflate_conv(block.conv2.weight.data, kT))

    bn2 = nn.BatchNorm3d(out_ch)
    _copy_bn(block.bn2, bn2)

    downsample_3d = None
    if block.downsample is not None:
        ds_conv = block.downsample[0]  # Conv2d(C_in, C_out, 1, 1)
        ds_bn = block.downsample[1]  # BatchNorm2d

        new_ds_conv = nn.Conv3d(
            ds_conv.in_channels,
            ds_conv.out_channels,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            bias=False,
        )
        new_ds_conv.weight.data.copy_(_inflate_shortcut(ds_conv.weight.data))

        new_ds_bn = nn.BatchNorm3d(ds_conv.out_channels)
        _copy_bn(ds_bn, new_ds_bn)

        downsample_3d = nn.Sequential(new_ds_conv, new_ds_bn)

    # relu is stateless — safe to share the same instance
    return BasicBlock3D(conv1, bn1, conv2, bn2, block.relu, downsample_3d)


# ── ResNet3DNetwork ───────────────────────────────────────────────────────────


class ResNet3DNetwork(nn.Module):
    r"""
    3D ResNet-18 backbone initialized from pretrained ImageNet weights via
    weight inflation.

    The 2D ImageNet weights are inflated to 3D as follows:
      - Stem conv: RGB channels are averaged then replicated across
        ``in_channels_level``; the temporal depth is replicated then
        divided by ``temporal_kernel_size_stem``.
      - Block 3×3 convs: temporal depth replicated and divided by 3.
      - 1×1 shortcut convs: unsqueezed to kT=1 (no temporal mixing).
      - BatchNorm: statistics copied directly.

    All spatial strides and the initial max-pool are removed so the full
    64×64 spatial resolution is maintained throughout the backbone.

    Parameters
    ----------
    in_channels_level : int, optional, default=4
        Number of level input variables (temperature, humidity, u, v).
    in_channels_aux : int, optional, default=2
        Number of auxiliary fields appended after level compression
        (land-sea mask, geopotential).
    n_levels : int, optional, default=7
        Number of pressure levels (depth dimension).
    hidden_dim : int, optional, default=64
        Width of the 2D fusion conv after level compression.
    temporal_kernel_size_stem : int, optional, default=3
        Temporal kernel size for the inflated stem convolution.
    pretrained : bool, optional, default=True
        Whether to load ImageNet-pretrained weights for inflation.
    """

    def __init__(
        self,
        in_channels_level: int = 4,
        in_channels_aux: int = 2,
        n_levels: int = 7,
        hidden_dim: int = 64,
        temporal_kernel_size_stem: int = 3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels_level = in_channels_level
        self.in_channels_aux = in_channels_aux
        self.n_levels = n_levels

        self.normalisation = InputNormalisation(
            mean=torch.tensor(_normalisation_mean),
            std=torch.tensor(_normalisation_std),
        )

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r2d = resnet18(weights=weights)

        # Inflated stem: Conv3d(C_in_level, 64, (kT_stem, 7, 7), stride=1)
        kT = temporal_kernel_size_stem
        self.stem_conv = nn.Conv3d(
            in_channels_level,
            64,
            kernel_size=(kT, 7, 7),
            stride=(1, 1, 1),
            padding=(kT // 2, 3, 3),
            bias=False,
        )
        self.stem_conv.weight.data.copy_(
            _inflate_first_conv(r2d.conv1.weight.data, in_channels_level, kT)
        )
        self.stem_bn = nn.BatchNorm3d(64)
        _copy_bn(r2d.bn1, self.stem_bn)
        self.stem_relu = nn.ReLU(inplace=True)
        # maxpool intentionally omitted to preserve spatial resolution

        # Inflated ResNet layers (temporal_depth=3 for all block 3×3 convs)
        self.layer1 = nn.Sequential(*[_inflate_basic_block(b, 3) for b in r2d.layer1])
        self.layer2 = nn.Sequential(*[_inflate_basic_block(b, 3) for b in r2d.layer2])
        self.layer3 = nn.Sequential(*[_inflate_basic_block(b, 3) for b in r2d.layer3])
        self.layer4 = nn.Sequential(*[_inflate_basic_block(b, 3) for b in r2d.layer4])

        # Collapse the level dimension: (B, 512, n_levels, H, W) → (B, 512, 1, H, W)
        self.compress_levels = nn.Conv3d(512, 512, kernel_size=(n_levels, 1, 1), padding=0)

        # 2D fusion of backbone features and auxiliary fields
        self.fusion = nn.Sequential(
            nn.Conv2d(512 + in_channels_aux, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        )

        self.output_layer = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        nn.init.normal_(self.output_layer.weight, std=1e-6)
        nn.init.constant_(self.output_layer.bias, 0.5)

    def forward(self, input_level: torch.Tensor, input_auxiliary: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters
        ----------
        input_level : torch.Tensor
            Pressure-level fields, shape ``(B, C_l, L, H, W)``.
        input_auxiliary : torch.Tensor
            Auxiliary fields, shape ``(B, C_a, H, W)``.

        Returns
        -------
        torch.Tensor
            Predictions of shape ``(B, 1, H, W)``.
        """
        BS, C, L, H, W = input_level.shape

        # Normalise: flatten levels → cat aux → normalize → split back
        flattened = input_level.reshape(BS, -1, H, W)
        aux_sliced = input_auxiliary[:, : self.in_channels_aux]
        norm_input = torch.cat([flattened, aux_sliced], dim=1).movedim(1, -1)
        norm_input = self.normalisation(norm_input).movedim(-1, 1)

        x_level = norm_input[:, : -self.in_channels_aux].reshape(BS, C, L, H, W)
        x_aux = norm_input[:, -self.in_channels_aux :]

        # 3D ResNet backbone
        x = self.stem_relu(self.stem_bn(self.stem_conv(x_level)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Level compression and aux fusion
        x = self.compress_levels(x).squeeze(2)  # (B, 512, H, W)
        x = self.fusion(torch.cat([x, x_aux], dim=1))  # (B, hidden_dim, H, W)
        return self.output_layer(x)  # (B, 1, H, W)


# ── ResNet3DModel ─────────────────────────────────────────────────────────────


class ResNet3DModel(BaseModel):
    r"""
    BaseModel wrapper for ResNet3DNetwork.

    Computes latitude-weighted MAE loss and auxiliary MSE / accuracy metrics.
    """

    def estimate_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        prediction = self.network(
            input_level=batch["input_level"],
            input_auxiliary=batch["input_auxiliary"],
        )
        prediction = prediction.clamp(0.0, 1.0)
        loss = (prediction - batch["target"]).abs()
        loss = (loss * self.lat_weights).mean()
        return {"loss": loss, "prediction": prediction}

    def estimate_auxiliary_loss(
        self, batch: Dict[str, torch.Tensor], outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        mse = (outputs["prediction"] - batch["target"]).pow(2)
        mse = (mse * self.lat_weights).mean()
        prediction_bool = (outputs["prediction"] > 0.5).float()
        target_bool = (batch["target"] > 0.5).float()
        accuracy = ((prediction_bool == target_bool).float() * self.lat_weights).mean()
        return {"mse": mse, "accuracy": accuracy}
