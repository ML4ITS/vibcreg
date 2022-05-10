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
from einops import repeat

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
        # self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=1)
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

        self.conv_att = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.linear_att = nn.Linear(out_channels, 1)

        self.layer_norm = nn.LayerNorm(out_channels)

    def _pad_x(self, x):
        out_size = x.shape[-1]
        in_size = x.shape[-1]
        padding = int(np.floor((out_size * 1 - in_size + self.kernel_size - 1) / 2))
        return F.pad(x, [padding, padding])

    def attend_out(self, out):
        """
        :param out (B, C, L)
        """
        att_out = self.conv_att(out)  # (B, C, L)
        att_out = att_out.transpose(1, 2)  # (B, L, C)
        att_out = F.leaky_relu(self.linear_att(att_out))  # (B, L, 1)
        att_out = F.softmax(att_out, dim=1)  # (B, L, 1)
        att_out = att_out * out.transpose(1, 2)  # (B, L, C)
        att_out = att_out.sum(dim=1)  # (B, C)
        return att_out

    def forward(self, x, detach_pooled_out=False):
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
        out = self.do1(out)  # (B, C, L)

        out_ = out.clone().detach() if detach_pooled_out else out  # (B, C, L)
        out_ = self.layer_norm(out_.transpose(1, 2)).transpose(1, 2)  # (B, C, L)
        gap_out = out_.mean(dim=-1)  # (B, C)
        gmp_out = out_.max(dim=-1).values  # (B, C)
        # att_out = self.attend_out(out)  # (B, C)
        amp_out = F.adaptive_max_pool1d(out_, max(1, int(0.5 * out_.shape[-1]))).mean(dim=-1)  # (B, C)
        comb_out = torch.cat((gap_out, gmp_out, amp_out), dim=-1)  # (B, 3C)

        return out, comb_out


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, input):
        return torch.transpose(input, self.dim1, self.dim2)


class OnionNet(nn.Module):
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
                    out_channels_enc = in_channels_enc * 1  #out_channels_enc = in_channels_enc * 2
                self.res_blocks.append(ResidualBlock(in_channels_enc, out_channels_enc, kernel_size_enc, stride, norm_layer_type_enc, dropout_rate_enc))
                in_channels_enc = out_channels_enc

        self.last_channels_enc = in_channels_enc

        # pooling layer(s)
        if pool_type == 'combined2':
            self.linear = nn.Linear(out_channels_enc, 1)

        # transformer to mix up onion-representations
        dim = 3*out_channels_enc
        self.attn_layers = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,
                                                                            nhead=1,
                                                                            dim_feedforward=2*dim,
                                                                            dropout=0.,
                                                                            activation='gelu',
                                                                            batch_first=True),
                                                 num_layers=1)
        self.norm_attn = nn.LayerNorm(dim)
        self.attn_linear = nn.Linear(2*dim, out_channels_enc)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, len(n_blocks_enc) + 1, dim))
        self.dropout_attn = nn.Dropout(0.)

        self.proj_gap = nn.Linear(out_channels_enc * len(self.res_blocks), out_channels_enc)
        self.proj_gmp = nn.Linear(out_channels_enc * len(self.res_blocks), out_channels_enc)
        # self.proj_att = nn.Linear(out_channels_enc * len(self.res_blocks), out_channels_enc)
        self.proj_amp = nn.Linear(out_channels_enc * len(self.res_blocks), out_channels_enc)

        self.layer_norms = nn.ModuleList([nn.LayerNorm(out_channels_enc) for _ in range(4)])

    @staticmethod
    def _flatten(x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return x

    def forward(self, x, return_concat_combined=True, projector_type='vibcreg', one_sided_partial_mask=0.):
        b = x.shape[0]
        n = len(self.res_blocks)

        out = self.first_block(x)
        comb_outs = torch.zeros((b, n, 3*self.last_channels_enc)).to(x.device)  # (B, N, 3C)
        for i, rb in enumerate(self.res_blocks):
            # out, comb_out = rb(out, detach_pooled_out=False if i == len(self.res_blocks)-1 else True)  # out: (B, C, L); comb_out: (B, 3C)
            # out, comb_out = rb(out, detach_pooled_out=False)
            out, comb_out = rb(out, detach_pooled_out=True)
            comb_outs[:, i, :] = comb_out

        if self.pool_type == 'combined2':
            # out = out.transpose(1, 2)  # (B, L, C)
            out = self.layer_norms[0](out.transpose(1, 2)).transpose(1, 2)  # (B, C, L)

            gap_out = out.mean(dim=-1)  # (B, C)
            gmp_out = out.max(dim=-1).values  # (B, C)
            amp_out = F.adaptive_max_pool1d(out, max(1, int(0.5 * out.shape[-1]))).mean(dim=-1)  # (B, C)

            gap_cout = comb_outs[:, :, :self.last_channels_enc]  # (B, N, C)
            gmp_cout = comb_outs[:, :, self.last_channels_enc: 2*self.last_channels_enc]  # (B, N, C)
            amp_cout = comb_outs[:, :, 2*self.last_channels_enc:]  # (B, N, C)

            # gap_cout = self.layer_norms[1](gap_cout)  # already done in the onion block
            # gmp_cout = self.layer_norms[2](gmp_cout)
            # amp_cout = self.layer_norms[3](amp_cout)

            gap_cout = torch.flatten(gap_cout, start_dim=1)  # (B, N*C)
            gmp_cout = torch.flatten(gmp_cout, start_dim=1)  # (B, N*C)
            amp_cout = torch.flatten(amp_cout, start_dim=1)  # (B, N*C)

            gap_cout = self.proj_gap(gap_cout)  # (B, C)
            gmp_cout = self.proj_gmp(gmp_cout)  # (B, C)
            amp_cout = self.proj_amp(amp_cout)  # (B, C)

            if return_concat_combined:
                return torch.cat((gap_out, gmp_out, amp_out, gap_cout, gmp_cout, amp_cout), dim=-1)  # (B, 3C)
                # return torch.cat((gap_cout, gmp_cout, amp_cout), dim=-1)  # (B, 3C)
            else:
                return (gap_out, gmp_out, amp_out), (gap_cout, gmp_cout, amp_cout)


if __name__ == '__main__':
    # generate a toy dataset
    batch_size = 4
    in_channels = 1
    H = 100  # horizon (length)
    x = torch.rand((batch_size, in_channels, H))

    # build a model
    resnet1d = OnionNet(in_channels, n_blocks_enc=(1, 1, 1, 1), pool_type='combined2', out_channels_enc=64)
    print("# resnet1d:\n", resnet1d, end='\n\n')
    print("last_channels_enc: ", resnet1d.last_channels_enc)

    # forward
    out = resnet1d(x)
    print("# output shape:", out.shape)  # (B, 3C) where C is out_channels_enc
