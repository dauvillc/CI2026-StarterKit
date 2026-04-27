#!/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit

# System modules
import logging
from typing import Any, Dict

# External modules
import torch
import torch.nn

from starter_kit.layers import InputNormalisation

# Internal modules
from starter_kit.model import BaseModel

main_logger = logging.getLogger(__name__)

r"""
The normalisation mean and std values are pre-computed from the training data.
As in the MLP, all pressure levels are collapsed into the channels dimension
and only the first two auxiliary fields (land sea mask and geopotential) are
used. For each of these 30 input features we compute the mean and std across
all spatial locations, weighted by the latitude weights, and averaged across
all time steps in the training set. These values are stored in the lists below
and used to initialise the InputNormalisation layer in the MLPNetwork.
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


class ConvBlock(torch.nn.Module):
    r"""
    A convolutional block with batch normalization and activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple, optional, default=3
        Size of the convolutional kernel.
    padding : int or tuple, optional, default=1
        Padding for the convolution.
    """

    def __init__(
        self,
        conv_layer: torch.nn.Conv3d,  # or 2D for non-level fields
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = conv_layer(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn = torch.nn.BatchNorm3d(out_channels)
        self.activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class CNNNetwork(torch.nn.Module):
    r"""
    3D CNN operating on pressure-level fields with auxiliary fields.

    Treats the pressure level dimension as depth for 3D convolutions,
    enabling the model to learn patterns across levels.

    Parameters
    ----------
    in_channels_level : int, optional, default=4
        Number of input variables (temperature, humidity, u, v).
    in_channels_aux : int, optional, default=5
        Number of auxiliary input channels.
    hidden_dim : int, optional, default=32
        Number of channels in hidden convolutional layers.
    n_levels : int, optional, default=7
        Number of pressure levels (depth dimension).
    n_blocks : int, optional, default=3
        Number of 3D convolutional blocks.
    """

    def __init__(
        self,
        in_channels_level: int = 4,
        in_channels_aux: int = 5,
        hidden_dim: int = 32,
        n_levels: int = 7,
        n_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels_level = in_channels_level
        self.in_channels_aux = in_channels_aux
        self.hidden_dim = hidden_dim
        self.n_levels = n_levels

        # Input normalisation for level fields
        self.normalisation = InputNormalisation(
            mean=torch.tensor(_normalisation_mean), std=torch.tensor(_normalisation_std)
        )

        # Initial 3D conv block for pressure-level fields
        self.initial_3d_conv = ConvBlock(
            torch.nn.Conv3d, in_channels_level, hidden_dim, kernel_size=3, padding=1
        )

        # Stack of 3D convolutional blocks
        conv_blocks = []
        for _ in range(n_blocks - 1):
            conv_blocks.append(
                ConvBlock(
                    torch.nn.Conv3d, hidden_dim, hidden_dim, kernel_size=3, padding=1
                )
            )
        self.conv_blocks = torch.nn.Sequential(*conv_blocks)

        # Compress level dimension to 1 while maintaining spatial resolution
        self.compress_levels = ConvBlock(
            torch.nn.Conv3d,
            hidden_dim,
            hidden_dim,
            kernel_size=(n_levels, 1, 1),
            padding=0,
        )

        # Fusion with auxiliary fields
        # After compression, we have (B, hidden_dim, 1, H, W), squeeze to (B, hidden_dim, H, W)
        # Then concatenate with auxiliary fields
        self.fusion_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                hidden_dim + in_channels_aux, hidden_dim, kernel_size=3, padding=1
            ),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.SiLU(),
        )

        # Final output layer
        self.output_layer = torch.nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0)
        torch.nn.init.normal_(self.output_layer.weight, std=1e-6)
        torch.nn.init.constant_(self.output_layer.bias, 0.5)

    def forward(
        self, input_level: torch.Tensor, input_auxiliary: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Forward pass: concatenate inputs, optionally normalise,
        then apply the MLP.

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

        ###### Normalise #######
        # We collapse all levels into the channel dimension
        flattened_input_level = input_level.reshape(BS, -1, H, W)
        # We only use the land sea mask and geopotential auxiliary fields
        sliced_auxiliary = input_auxiliary[:, : self.in_channels_aux]

        # Concatenate the level and auxiliary fields
        flattened_input_level = torch.cat(
            [flattened_input_level, sliced_auxiliary], dim=1
        )

        # Move the feature dimension to the end for normalisation and MLP
        norm_input = flattened_input_level.movedim(1, -1)

        # Apply input normalisation
        norm_input = self.normalisation(norm_input)

        # Undo flattening and channel move for convolutional processing
        norm_input = norm_input.movedim(-1, 1)
        x_level = norm_input[:, :-2].reshape(BS, C, L, H, W)
        x_aux = norm_input[:, -2:]

        # Apply initial 3D conv block
        x = self.initial_3d_conv(x_level)  # (B, hidden_dim, L, H, W)

        # Apply stacked 3D conv blocks
        x = self.conv_blocks(x)  # (B, hidden_dim, L, H, W)

        # Compress level dimension
        x = self.compress_levels(x)  # (B, hidden_dim, 1, H, W)
        x = x.squeeze(2)  # (B, hidden_dim, H, W)

        # Fuse with auxiliary fields
        # Only use first 2 auxiliary fields (land-sea mask, geopotential)
        # aux_sliced = input_auxiliary[:, :2]  # (B, 2, H, W)
        x = torch.cat([x, x_aux], dim=1)  # (B, hidden_dim+2, H, W)

        # Fusion convolution
        x = self.fusion_conv(x)  # (B, hidden_dim, H, W)

        # Output layer
        output = self.output_layer(x)  # (B, 1, H, W)

        return output


class CNNModel(BaseModel):
    r"""
    Model wrapper for a 3D CNN network.

    This class delegates forward execution to a 3D CNN network and
    computes mean absolute error loss together with auxiliary metrics.
    """

    def estimate_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        r"""
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
            ``loss`` is the weighted mean absolute error and ``prediction``
            is the model output clamped to ``[0, 1]``.
        """
        prediction = self.network(
            input_level=batch["input_level"], input_auxiliary=batch["input_auxiliary"]
        )
        prediction = prediction.clamp(0.0, 1.0)
        loss = (prediction - batch["target"]).abs()
        loss = loss * self.lat_weights
        loss = loss.mean()
        return {"loss": loss, "prediction": prediction}

    def estimate_auxiliary_loss(
        self, batch: Dict[str, torch.Tensor], outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        r"""
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
        """
        mse = (outputs["prediction"] - batch["target"]).pow(2)
        mse = (mse * self.lat_weights).mean()
        prediction_bool = (outputs["prediction"] > 0.5).float()
        target_bool = (batch["target"] > 0.5).float()
        accuracy = (prediction_bool == target_bool).float()
        accuracy = (accuracy * self.lat_weights).mean()
        return {"mse": mse, "accuracy": accuracy}
