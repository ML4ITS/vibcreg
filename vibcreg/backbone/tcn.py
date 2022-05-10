import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, input):
        return torch.transpose(input, self.dim1, self.dim2)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, pool_type='gap', **kwargs):
        super(TemporalConvNet, self).__init__()
        self.pool_type = pool_type

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.last_channels_enc = out_channels

        # pooling layer(s)
        if pool_type == 'gap':
            self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        elif pool_type == 'gmp':
            self.global_maxpool = nn.AdaptiveMaxPool1d(1)
        elif pool_type == 'att':
            self.att = nn.Sequential(Transpose(1, 2),
                                     nn.Linear(out_channels, 1),
                                     nn.ReLU()
                                     )
        elif pool_type == 'combined':
            self.att = nn.Sequential(
                Transpose(1, 2),
                nn.Linear(out_channels, 1),
                nn.ReLU()
            )
        elif pool_type == 'combined2':
            self.linear = nn.Linear(out_channels, 1)

    @staticmethod
    def _flatten(x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return x

    def forward(self, x, return_concat_combined=True, projector_type='vibcreg', one_sided_partial_mask=0.):
        out = self.network(x)

        # pooling
        if self.pool_type == 'gap':
            out = self.global_avgpool(out)
            out = self._flatten(out)
            return out
        elif self.pool_type == 'gmp':
            out = self.global_maxpool(out)
            out = self._flatten(out)
            return out
        elif self.pool_type == 'att':
            att_score = self.att(out)  # (N, L, 1)
            att_score = torch.softmax(att_score, dim=1)
            out = torch.transpose(out, 1, 2)  # (N, L, C)

            out = out * att_score  # (N, L, C)
            out = torch.sum(out, dim=1)  # (N, C)

            return out
        elif self.pool_type == 'combined':
            out_gap = torch.mean(out, dim=2)  # (N, C)
            out_gmp = torch.max(out, dim=2).values  # (N, C)

            att_score = self.att(out)  # (N, L, 1)
            att_score = torch.softmax(att_score, dim=1)
            out_att = torch.transpose(out, 1, 2)  # (N, L, C)
            out_att = out_att * att_score  # (N, L, C)
            out_att = torch.sum(out_att, dim=1)  # (N, C)

            # out = torch.cat((out_gap, out_gmp, out_att), dim=-1)  # (N, 3*C)

            if return_concat_combined:
                return torch.cat((out_gap, out_gmp, out_att), dim=-1)
            else:
                return out_gap, out_gmp, out_att

        elif self.pool_type == 'combined2':
            out = out.transpose(1, 2)  # (N, L, C)

            while True:
                ind = np.random.rand(out.shape[1]) > one_sided_partial_mask
                if True in ind:
                    break
            out = out[:, ind, :]

            out_gap = torch.mean(out, dim=1)  # (N, C)
            out_gmp = torch.max(out, dim=1).values  # (N, C)

            att_score = torch.relu(self.linear(out))  # (N, L, 1)
            out_att = out * att_score  # (N, L, C)
            out_att = torch.sum(out_att, dim=1)  # (N, C)

            if return_concat_combined:
                return torch.cat((out_gap, out_gmp, out_att), dim=-1)
            else:
                return out_gap, out_gmp, out_att


if __name__ == '__main__':
    import torch
    # toy dataset
    B, C, L = 4, 1, 100
    X = torch.rand((B, C, L))

    # model
    model = TemporalConvNet(C, [32, 32, 32, 32, ])
    print(model)

    # forward
    out = model(X)
    print(out.shape)
