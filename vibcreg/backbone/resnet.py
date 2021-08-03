"""
Define `ResNet1D` which consists of `FirstBlock` and `ResidualBlock`.

# References:
[1] K. He et al., 2016, "Deep residual learning for image recognition" <br>
[2] geekfeiw, "Multi-Scale-1D-ResNet", [github_link](https://github.com/geekfeiw/Multi-Scale-1D-ResNet/blob/master/figs/network.png)<br>
[3] hsd1503, "resnet1d", [github_link](https://github.com/hsd1503/resnet1d/blob/master/resnet1d.py)<br>
[4] kuangliu, "pytorch-cifar/models/resnet.py", [github_link](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import relu

from vibcreg.normalization.iter_norm import IterNorm


def normalization_layer(norm_layer_type, num_channels, dim, num_groups_IterN=64):
    """
    dim: #dimension of data
    """
    if norm_layer_type == "BatchNorm":
        return nn.BatchNorm1d(num_channels)
    elif norm_layer_type == 'non_affine_BatchNorm':
        return nn.BatchNorm1d(num_channels, affine=False, eps=0.)
    elif norm_layer_type == "LayerNorm":
        return nn.GroupNorm(1, num_channels)
    elif norm_layer_type == "GroupNorm":
        return nn.GroupNorm(32, num_channels)  # 32 is a default value from the original paper.
    elif norm_layer_type == 'IterNorm':
        return IterNorm(num_channels, num_groups=num_groups_IterN, T=5, dim=dim, affine=True)
    else:
        raise ValueError(f"unavailable 'norm_layer_type': {norm_layer_type}")


class FirstBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer_type, dropout_rate):
        super().__init__()
        self.kernel_size = kernel_size

        # define layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.nl1 = normalization_layer(norm_layer_type, out_channels, dim=3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.do1 = nn.Dropout(dropout_rate)

    def _pad_x(self, x):
        out_size = x.shape[-1]
        in_size = x.shape[-1]
        padding = int(np.floor((out_size * 1 - in_size + self.kernel_size - 1) / 2))
        return F.pad(x, (padding, padding))

    def forward(self, x):
        out = self._pad_x(x)
        out = self.conv1(out)
        out = self.nl1(out)
        out = relu(out)
        out = self.maxpool1(out)
        out = self.do1(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer_type, dropout_rate):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # define layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.nl1 = normalization_layer(norm_layer_type, out_channels, dim=3)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1)
        self.nl2 = normalization_layer(norm_layer_type, out_channels, dim=3)
        self.do1 = nn.Dropout(dropout_rate)

        if self.stride != 1:
            self.conv_1x1 = nn.Conv1d(in_channels, out_channels, 1, stride)

    def _pad_x(self, x):
        out_size = x.shape[-1]
        in_size = x.shape[-1]
        padding = int(np.floor((out_size * 1 - in_size + self.kernel_size - 1) / 2))
        return F.pad(x, [padding, padding])

    def forward(self, x):
        out = self._pad_x(x)
        out = self.conv1(out)
        out = self.nl1(out)
        out = relu(out)

        out = self._pad_x(out)
        out = self.conv2(out)
        out = self.nl2(out)

        if x.shape == out.shape:
            out = out + x
            out = relu(out)
        else:
            x = self.conv_1x1(x)
            out = out + x
            out = relu(out)
        out = self.do1(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, in_channels, n_blocks=[1, 1, 1, 1], out_channels=64, kernel_size=3, norm_layer_type="BatchNorm", dropout_rate=0.):
        """
        :param in_channels: dimension of an input; e.g., 1 for the UCR datasets, and 12 for the PTB-XL.
        :param n_blocks: e.g., [3, 4, 6, 3]; The same residual block is repeated 3 times -> dimension increase -> another same residual block is repeated 4  times -> ...
        :param out_channels: dimension of an output after the first block; the dimension automatically doubles after every set of the same residual blocks while the input dimension decreases by double.
        :param kernel_size: kernel size in residual blocks
        :param norm_layer_type: type of the normalization layers
        :param dropout_rate: rate for `Dropout`. If 0, `Dropout` is not used.
        """
        super().__init__()

        # define blocks
        self.first_block = FirstBlock(in_channels, out_channels, 7, 2, norm_layer_type, dropout_rate)

        self.res_blocks = nn.ModuleList()
        in_channels = out_channels
        for i, n_block in enumerate(n_blocks):
            for j in range(n_block):
                stride = 2 if ((i != 0) and (j == 0)) else 1
                if (i != 0) and (j == 0):
                    out_channels = in_channels * 2
                self.res_blocks.append(ResidualBlock(in_channels, out_channels, kernel_size, stride, norm_layer_type, dropout_rate))
                in_channels = out_channels

        self.out_channels_backbone = in_channels

        # define global average pooling layer
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)

    @staticmethod
    def _flatten(x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return x

    def forward(self, x):
        out = self.first_block(x)
        for rb in self.res_blocks:
            out = rb(out)

        out = self.global_avgpool(out)
        out = self._flatten(out)
        return out


if __name__ == "__main__":
    # build a model
    resnet1d = ResNet1D(in_channels=1)
    print("# resnet1d:\n", resnet1d, end='\n\n')
    print("out_channels_backbone: ", resnet1d.out_channels_backbone)

    # generate a toy dataset
    batch_size = 32
    in_channels = 1
    H = 224  # horizon (length)
    X = torch.rand((batch_size, in_channels, H))

    # forward
    out = resnet1d(X)
    print("# output shape:", out.shape)
