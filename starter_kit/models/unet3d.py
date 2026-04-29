#!/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit

import logging
from typing import Any, Dict

import torch
import torch.nn
import torch.nn.functional as F

from starter_kit.layers import InputNormalisation
from starter_kit.model import BaseModel

main_logger = logging.getLogger(__name__)

r"""
Normalisation mean/std pre-computed from training data (same as cnn.py):
28 level features (4 vars × 7 levels) + 2 auxiliary fields.
"""

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


class DoubleConv3D(torch.nn.Module):
    r"""Two consecutive Conv3d → BN3d → SiLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.SiLU(),
            torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv2D(torch.nn.Module):
    r"""Two consecutive Conv2d → BN2d → SiLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3DNetwork(torch.nn.Module):
    r"""
    Hybrid 3D-encoder / 2D-decoder U-Net for cloud cover prediction.

    The encoder uses 3D convolutions treating pressure levels as the depth
    dimension, storing skip connections at each scale. After the bottleneck,
    the level dimension is collapsed via mean pooling and the decoder operates
    purely in 2D, using the compressed skip connections.

    Parameters
    ----------
    in_channels_level : int, optional, default=4
        Number of input variables per pressure level (temp, humidity, u, v).
    in_channels_aux : int, optional, default=2
        Number of auxiliary input channels used for normalisation and fusion.
    base_channels : int, optional, default=32
        Base channel width; doubles at each encoder stage.
    n_levels : int, optional, default=7
        Number of pressure levels (depth dimension).
    """

    def __init__(
        self,
        in_channels_level: int = 4,
        in_channels_aux: int = 2,
        base_channels: int = 32,
        n_levels: int = 7,
    ) -> None:
        super().__init__()
        self.in_channels_level = in_channels_level
        self.in_channels_aux = in_channels_aux
        self.n_levels = n_levels
        C = base_channels

        n_norm_channels = in_channels_level * n_levels + in_channels_aux
        self.normalisation = InputNormalisation(
            mean=torch.tensor(_normalisation_mean[:n_norm_channels]),
            std=torch.tensor(_normalisation_std[:n_norm_channels]),
        )

        # Encoder
        self.enc1 = DoubleConv3D(in_channels_level, C)
        self.enc2 = DoubleConv3D(C, C * 2)
        self.enc3 = DoubleConv3D(C * 2, C * 4)
        self.bottleneck = DoubleConv3D(C * 4, C * 8)
        # Spatial-only pooling — level dim preserved throughout encoder
        self.pool = torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Decoder
        self.dec3 = DoubleConv2D(C * 8 + C * 4, C * 4)
        self.dec2 = DoubleConv2D(C * 4 + C * 2, C * 2)
        self.dec1 = DoubleConv2D(C * 2 + C, C)

        # Auxiliary fusion after full-resolution decoder output
        self.fusion = torch.nn.Sequential(
            torch.nn.Conv2d(C + in_channels_aux, C, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(C),
            torch.nn.SiLU(),
        )

        self.output_layer = torch.nn.Conv2d(C, 1, kernel_size=1)
        torch.nn.init.normal_(self.output_layer.weight, std=1e-6)
        torch.nn.init.constant_(self.output_layer.bias, 0.5)

    def forward(
        self, input_level: torch.Tensor, input_auxiliary: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Parameters
        ----------
        input_level : torch.Tensor
            Shape ``(B, C_l, L, H, W)``.
        input_auxiliary : torch.Tensor
            Shape ``(B, C_a, H, W)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1, H, W)``.
        """
        BS, C, L, H, W = input_level.shape

        # --- Normalisation (same strategy as cnn.py) ---
        x_flat = input_level.reshape(BS, -1, H, W)
        x_aux_slice = input_auxiliary[:, : self.in_channels_aux]
        norm_in = torch.cat([x_flat, x_aux_slice], dim=1).movedim(1, -1)
        norm_in = self.normalisation(norm_in).movedim(-1, 1)

        x_level = norm_in[:, : C * L].reshape(BS, C, L, H, W)
        x_aux = norm_in[:, C * L :]  # (B, in_channels_aux, H, W)

        # --- Encoder (3D) ---
        s1 = self.enc1(x_level)          # (B, C, L, 64, 64)
        s2 = self.enc2(self.pool(s1))    # (B, 2C, L, 32, 32)
        s3 = self.enc3(self.pool(s2))    # (B, 4C, L, 16, 16)
        b = self.bottleneck(self.pool(s3))  # (B, 8C, L, 8, 8)

        # --- Collapse level dim → 2D (mean over L) ---
        b2d = b.mean(dim=2)              # (B, 8C, 8, 8)
        s3_2d = s3.mean(dim=2)           # (B, 4C, 16, 16)
        s2_2d = s2.mean(dim=2)           # (B, 2C, 32, 32)
        s1_2d = s1.mean(dim=2)           # (B, C, 64, 64)

        # --- Decoder (2D) ---
        x = F.interpolate(b2d, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, s3_2d], dim=1))   # (B, 4C, 16, 16)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, s2_2d], dim=1))   # (B, 2C, 32, 32)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, s1_2d], dim=1))   # (B, C, 64, 64)

        # --- Auxiliary fusion and output ---
        x = self.fusion(torch.cat([x, x_aux], dim=1))  # (B, C, 64, 64)
        return self.output_layer(x)                     # (B, 1, 64, 64)


class UNet3DModel(BaseModel):
    r"""Model wrapper for the hybrid 3D-encoder / 2D-decoder U-Net."""

    def estimate_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        prediction = self.network(
            input_level=batch["input_level"], input_auxiliary=batch["input_auxiliary"]
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
