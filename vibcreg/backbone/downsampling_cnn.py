"""
`Downsampling CNN` denotes a CNN model where the downsampling is done by striding.
Note that there is no skip-connection. It is just 'ordinary CNN architecture' + 'downsampling by striding'.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import relu

from vibcreg.backbone.resnet import normalization_layer


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer_type):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        # define layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, self.kernel_size, self.stride)
        self.nl1 = normalization_layer(norm_layer_type, out_channels, dim=3)

        self.padding = None

    def _pad_x(self, x):
        """Pad 'x' before being fed to a conv layer."""
        out_size = x.shape[-1]
        in_size = x.shape[-1]
        stride = 1  # to allow the strided downsampling.
        self.padding = int(np.floor((out_size * stride - in_size + self.kernel_size - 1) / 2))
        return F.pad(x, [self.padding, self.padding])

    def forward(self, x):
        x = self._pad_x(x)
        out = self.conv1(x)
        out = relu(self.nl1(out))
        return out


class DownsamplingCNN(nn.Module):
    """Downsampling CNN encoder used for CPC"""

    def __init__(self, downsampling_layers_cpc=(4, 3, 3, 2), enc_hid_channels_cpc=512, norm_layer_type_cpc="BatchNorm", **kwargs):
        super().__init__()
        self.downsampling_layers = downsampling_layers_cpc
        self.enc_hid_channels_cpc = enc_hid_channels_cpc

        # define downsampling blocks
        self.downsample_blocks = nn.ModuleList()
        in_channels = 12  # n_channels of your data
        for i, stride in enumerate(self.downsampling_layers):
            out_channels = self.enc_hid_channels_cpc
            kernel_size = 2 * stride
            downsample_block = DownsamplingBlock(in_channels, out_channels, kernel_size, stride, norm_layer_type_cpc)
            self.downsample_blocks.append(downsample_block)
            in_channels = out_channels

        self.downsampling_factor = self._compute_downsampling_factor()
        self.interval_original_space = self._compute_interval_on_original_space()
        self.H_original_space = self._compute_horizon_on_original_space()

    def _compute_downsampling_factor(self, ):
        """
        [UPDATES] It turns out `downsampling_factor` is equal to `interval_original_space`.
        """
        in_size_origin = 20480  # any large number is okay.

        # update `self.padding` in `downsample_block`s
        x = torch.zeros(1, 12, in_size_origin)
        _ = self.forward(x)

        # compute
        in_size = in_size_origin
        for downsample_block in self.downsample_blocks:
            kernel_size = downsample_block.kernel_size
            padding = downsample_block.padding
            stride = downsample_block.stride

            out_size = np.floor((in_size - kernel_size + 2 * padding) / stride + 1)
            in_size = out_size
        return in_size_origin / out_size

    def _compute_interval_on_original_space(self):
        """
        It computes an interval on the original space.

        # Detailed Explanation:
        - An original sequence is reduced along its sequential length dimension via the encoder.
        - One step in the representation space is equivalent to `n`-steps in the original space.
        - This method computes `n`.
        """
        return np.prod(self.downsampling_layers)

    def _compute_horizon_on_original_space(self):
        """
        # Detailed Explanation:
        - An original sequence is reduced along its sequential length dimension via the encoder.
        - 1 step-long Horizon (H) on the representation space is equivalent to `n` step-long H on the original space.
        - This method computes H on the original space.
        """
        Hr = 1  # `H` on the representation space

        out_size = Hr
        for downsample_block in self.downsample_blocks[::-1]:
            in_size = (out_size - 1) * downsample_block.stride + downsample_block.kernel_size  # `padding=0`; we don't consider padding for this computation.
            out_size = in_size
        Ho = out_size  # `H` on the origianl space
        return Ho

    def forward(self, x):
        out = x
        for downsample_block in self.downsample_blocks:
            out = downsample_block(out)
        return out