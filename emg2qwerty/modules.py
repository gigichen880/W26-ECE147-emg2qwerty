# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn


def _conv1d_output_lengths(
    lengths: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    """Return the valid output lengths after a 1D convolution."""
    return ((lengths + (2 * padding) - (dilation * (kernel_size - 1)) - 1) // stride) + 1


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class ResBlock1d(nn.Module):
    """1D Residual Block with skip connection for stable training.
    
    Features skip connections and GELU activation for improved gradient flow
    and preventing dead neurons.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size for convolution (default: 5)
        stride (int): Stride for convolution (default: 1)
        padding (int): Padding for convolution (default: 2)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5,
                 stride: int = 1, padding: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        
        # Projection for skip connection if channels or stride change
        self.proj = None
        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv(x)
        out = self.bn(out)
        out = self.gelu(out)
        
        # Apply projection if needed for skip connection
        if self.proj is not None:
            identity = self.proj(x)
        
        # Add skip connection
        out = out + identity
        return out

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return _conv1d_output_lengths(
            lengths=lengths,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )


class CnnEncoder(nn.Module):
    """
    Pure temporal Conv1d encoder for CTC-based sequence modeling.

    Input: (T, N, num_features)
    Output: (T_out, N, conv_channels[-1])
    """

    def __init__(
        self,
        num_features: int,
        latent_dim: int = 256,
        conv_channels: Sequence[int] = (512, 1024, 1024),
        kernel_sizes: Sequence[int] = (7, 5, 5),
        strides: Sequence[int] = (2, 2, 2),
        dropout_probs: Sequence[float] = (0.2, 0.2, 0.3),
    ) -> None:
        super().__init__()

        assert len(conv_channels) > 0
        assert len(conv_channels) == len(kernel_sizes) == len(strides) == len(
            dropout_probs
        )

        self.projection = nn.Linear(num_features, latent_dim)

        blocks: list[nn.Module] = []
        residual_blocks: list[ResBlock1d] = []
        in_channels = latent_dim
        for out_channels, kernel_size, stride, dropout in zip(
            conv_channels, kernel_sizes, strides, dropout_probs
        ):
            block = ResBlock1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
            residual_blocks.append(block)
            blocks.extend([block, nn.Dropout(dropout)])
            in_channels = out_channels

        self.residual_blocks = nn.ModuleList(residual_blocks)
        self.enc = nn.Sequential(*blocks)
        self.output_dim = conv_channels[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # (T, N, C) -> (T, N, latent_dim)
        x = self.projection(inputs)

        # (T, N, latent_dim) -> (N, latent_dim, T)
        x = x.permute(1, 2, 0)
        x = self.enc(x)

        # (N, C_out, T_out) -> (T_out, N, C_out)
        return x.permute(2, 0, 1)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = input_lengths.to(dtype=torch.long)
        for block in self.residual_blocks:
            lengths = block.output_lengths(lengths)
        return lengths.to(dtype=input_lengths.dtype)


class CnnRnnEncoder(nn.Module):
    """
    CNN + BiLSTM encoder for EMG feature extraction.
    Architecture: Linear Projection -> ResNet-style Conv1D -> BiLSTM
    
    Input: (T, N, num_features) - flattened EMG features (matching TDSConvEncoder format)
    Output: (T, N, hidden_size * 2) - encoded features
    
    Key improvements:
    - Linear projection bottleneck compresses high-dimensional input (e.g., 700 dims)
      to a stable latent space (256 dims) before convolutions
    - Residual blocks with skip connections ensure stable gradient flow
    - GELU activation instead of ReLU to prevent dead neurons
    - 8x temporal reduction via 3 stride-2 layers (balanced for CTC alignment)
    """
    
    def __init__(self, num_features: int, hidden_size: int = 128,
                 num_layers: int = 2, latent_dim: int = 256):
        super(CnnRnnEncoder, self).__init__()

        self.cnn = CnnEncoder(
            num_features=num_features,
            latent_dim=latent_dim,
        )

        # Bi-directional LSTM for context modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn.output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (T, N, num_features) - TNC format
        
        Returns:
            features: (T, N, hidden_size * 2) - TNC format
        """
        x = self.cnn(inputs)

        # (T_reduced, N, C) -> (N, T_reduced, C)
        x = x.permute(1, 0, 2)

        # BiLSTM: (N, T_reduced, 1024) -> (N, T_reduced, hidden_size*2)
        lstm_out, _ = self.lstm(x)

        # Convert back to TNC format: (N, T_reduced, hidden_size*2) -> (T_reduced, N, hidden_size*2)
        features = lstm_out.permute(1, 0, 2)

        return features

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return self.cnn.output_lengths(input_lengths)
