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


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, input):
        return torch.transpose(input, self.dim1, self.dim2)


class ResNet1D(nn.Module):
    def __init__(self,
                 in_channels_enc,
                 n_blocks_enc=(1, 1, 1, 1),
                 out_channels_enc=64,
                 kernel_size_enc=3,
                 norm_layer_type_enc="BatchNorm",
                 dropout_rate_enc=0.,
                 pool_type='gap',
                 **kwargs):
        """
        :param in_channels_enc: dimension of an input; e.g., 1 for the UCR datasets, and 12 for the PTB-XL.
        :param n_blocks_enc: e.g., [3, 4, 6, 3]; The same residual block is repeated 3 times -> dimension increase -> another same residual block is repeated 4  times -> ...
        :param out_channels_enc: dimension of an output after the first block; the dimension automatically doubles after every set of the same residual blocks while the input dimension decreases by double.
        :param kernel_size_enc: kernel size in residual blocks
        :param norm_layer_type_enc: type of the normalization layers
        :param dropout_rate_enc: rate for `Dropout`. If 0, `Dropout` is not used.
        """
        super().__init__()
        self.pool_type = pool_type

        # define blocks
        self.first_block = FirstBlock(in_channels_enc, out_channels_enc, 7, 2, norm_layer_type_enc, dropout_rate_enc)

        self.res_blocks = nn.ModuleList()
        in_channels_enc = out_channels_enc
        for i, n_block in enumerate(n_blocks_enc):
            for j in range(n_block):
                stride = 2 if ((i != 0) and (j == 0)) else 1
                if (i != 0) and (j == 0):
                    out_channels_enc = in_channels_enc * 2
                self.res_blocks.append(ResidualBlock(in_channels_enc, out_channels_enc, kernel_size_enc, stride, norm_layer_type_enc, dropout_rate_enc))
                in_channels_enc = out_channels_enc

        self.last_channels_enc = in_channels_enc

        # pooling layer(s)
        if pool_type == 'gap':
            self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        elif pool_type == 'gmp':
            self.global_maxpool = nn.AdaptiveMaxPool1d(1)
        elif pool_type == 'att' or pool_type == 'combined':
            # self.att = nn.Sequential(Transpose(1, 2),
            #                          nn.Linear(out_channels_enc, 1),
            #                          nn.ReLU()
            #                          )
            h_size = out_channels_enc
            bidirectional = True
            num_layers = 1
            D = 2 if bidirectional else 1
            self.gru = nn.GRU(out_channels_enc, hidden_size=h_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
            self.linear_gru = nn.Linear(2*h_size, h_size)
            self.h_0 = None
            self.init_states = lambda batch_size: torch.zeros(D * num_layers, batch_size, out_channels_enc)  # D*n_layers, batch_size, h_size

            h_size = out_channels_enc // 2
            bidirectional = True
            num_layers = 1
            D = 2 if bidirectional else 1
            self.gru_p = nn.GRU(out_channels_enc, hidden_size=h_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
            self.h_0_p = None
            self.init_states_p = lambda batch_size: torch.zeros(D*num_layers, batch_size, h_size)  # D*n_layers, batch_size, h_size

            self.linear = nn.Linear(D*h_size, 1)

        elif pool_type == 'combined2':
            self.linear = nn.Linear(out_channels_enc, 1)

    @staticmethod
    def _flatten(x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return x

    def forward(self, x, return_concat_combined=True, projector_type='vibcreg'):
        out = self.first_block(x)
        for rb in self.res_blocks:
            out = rb(out)  # (N, C, L)

        if self.pool_type == 'gap':
            out = self.global_avgpool(out)
            out = self._flatten(out)
            return out
        elif self.pool_type == 'gmp':
            out = self.global_maxpool(out)
            out = self._flatten(out)
            return out
        elif self.pool_type == 'att':
            out = out.transpose(1, 2)  # (N, L, C)

            if projector_type == 'simclr':
                # att_score = att_score.detach()
                out = torch.mean(out, dim=1)  # (N, C)
            else:
                self.h_0 = self.init_states(out.shape[0]).to(out.device)
                att_score, hn = self.gru(out, self.h_0)  # (N, L, D)
                att_score = torch.relu(self.linear(att_score))  # (N, L, 1)
                att_score = torch.softmax(att_score, dim=1)
                out = out * att_score  # (N, L, C)
                out = torch.sum(out, dim=1)  # (N, C)
            return out
        elif self.pool_type == 'combined':
            out = out.transpose(1, 2)  # (N, L, C)
            self.h_0 = self.init_states(out.shape[0]).to(out.device)
            out, hn = self.gru(out, self.h_0)  # (N, L, C)
            out = self.linear_gru(out)

            out_gap = torch.mean(out, dim=1)  # (N, C)
            out_gmp = torch.max(out, dim=1).values  # (N, C)

            # att_score = self.att(out)  # (N, L, 1)
            # att_score = torch.softmax(att_score, dim=1)
            # out_att = torch.transpose(out, 1, 2)  # (N, L, C)
            # out_att = out_att * att_score  # (N, L, C)
            # out_att = torch.sum(out_att, dim=1)  # (N, C)
            # out = torch.cat((out_gap, out_gmp, out_att), dim=-1)  # (N, 3*C)

            self.h_0_p = self.init_states_p(out.shape[0]).to(out.device)
            # att_score, hn = self.gru_p(out, self.h_0_p)  # (N, L, D)
            att_score, hn = self.gru_p(out.detach(), self.h_0_p)  # (N, L, D)
            att_score = torch.relu(self.linear(att_score))  # (N, L, 1)
            att_score = torch.softmax(att_score, dim=1)
            out_att = out * att_score  # (N, L, C)
            out_att = torch.sum(out_att, dim=1)  # (N, C)

            if return_concat_combined:
                return torch.cat((out_gap, out_gmp, out_att), dim=-1)
            else:
                return out_gap, out_gmp, out_att
        elif self.pool_type == 'combined2':
            out = out.transpose(1, 2)  # (N, L, C)

            out_gap = torch.mean(out, dim=1)  # (N, C)
            out_gmp = torch.max(out, dim=1).values  # (N, C)

            att_score = torch.relu(self.linear(out))  # (N, L, 1)
            out_att = out * att_score  # (N, L, C)
            out_att = torch.sum(out_att, dim=1)  # (N, C)

            if return_concat_combined:
                return torch.cat((out_gap, out_gmp, out_att), dim=-1)
            else:
                return out_gap, out_gmp, out_att


if __name__ == "__main__":
    # build a model
    resnet1d = ResNet1D(in_channels_enc=1, n_blocks_enc=(4, 4), pool_type='att')
    print("# resnet1d:\n", resnet1d, end='\n\n')
    print("last_channels_enc: ", resnet1d.last_channels_enc)

    # generate a toy dataset
    batch_size = 32
    in_channels = 1
    H = 750  # horizon (length)
    X = torch.rand((batch_size, in_channels, H))

    # forward
    out = resnet1d(X)
    print("# output shape:", out.shape)
