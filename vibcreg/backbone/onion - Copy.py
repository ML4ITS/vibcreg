import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import numpy as np


class OnionNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.stride = stride

        # define layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # self.norm1 = nn.GroupNorm(1, out_channels)  # layer norm
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

        if stride > 1:
            self.linear = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)

        # attention
        # self.conv_att = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv_att = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1)
        self.linear_att = nn.Linear(out_channels, 1)
        # self.pool = nn.Linear(2*out_channels, out_channels)

    def attend_out(self, out):
        """
        attend out
        :param out (B, C, L)
        """
        att_out = self.conv_att(out)  # (B, C, L)
        att_out = att_out.transpose(1, 2)  # (B, L, C)
        att_out = F.leaky_relu(self.linear_att(att_out))  # (B, L, 1)
        att_out = F.softmax(att_out, dim=1)  # (B, L, 1)
        att_out = att_out * out.transpose(1, 2)  # (B, L, C)
        att_out = att_out.sum(dim=1)  # (B, C)
        return att_out

    def forward(self, x):
        out = self.act(self.norm1(self.conv1(x)))

        if self.stride == 1:
            out = out + x
        elif self.stride > 1:
            out = out + self.linear(x)

        out = self.act(out)  # (B, C, L)

        # get attended `out`
        att_out = self.attend_out(out.detach())  # (B, C)
        # att_out = out.mean(dim=-1)  # (B, C)
        # att_out = out.max(dim=-1).values  # (B, C)

        # att_out = torch.cat((out.mean(dim=-1), out.max(dim=-1).values), dim=-1)  # (B, 2C)
        # att_out = self.pool(att_out)  # (B, C)

        return out, att_out


class OnionNet(nn.Module):
    def __init__(self,
                 in_channels,
                 dim=32,
                 strides=(2, 2, 1, 1),
                 nhead=2,
                 num_attn_layers=2,
                 dropout_attn=0.,
                 emb_dropout=0.,
                 **kwargs,
                 ):
        super().__init__()
        self.dim = dim
        self.strides = strides
        self.last_channels_enc = dim

        # onion blocks
        self.blocks = nn.ModuleList()
        for stride in strides:
            self.blocks.append(OnionNetBlock(in_channels, dim, stride))
            in_channels = dim

        # transformer to mix up onion-representations
        self.attn_layers = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,
                                                                            nhead=nhead,
                                                                            dim_feedforward=min(dim*4, 2048),
                                                                            dropout=dropout_attn,
                                                                            activation='gelu',
                                                                            batch_first=True),
                                                 num_layers=num_attn_layers)
        self.linear_attn = nn.Linear(dim, 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, len(strides) + 1, dim))
        self.dropout_attn = nn.Dropout(emb_dropout)
        self.norm_attn = nn.LayerNorm(dim)

        self.mask_token = nn.Parameter(torch.randn(1, dim, 1))

    # def forward(self, x, **kwargs):
    #     """
    #     :param x (B, C, L)
    #     """
    #     b = x.shape[0]
    #     n = len(self.blocks)
    #     out = x
    #
    #     att_outs = torch.zeros(b, len(self.blocks), self.dim).to(x.device)  # (B, N, C); store representations from each block
    #     for i, block in enumerate(self.blocks):
    #         out, att_out = block(out)  # out:(B, C, L), att_out:(B, C)
    #         att_outs[:, i, :] = att_out
    #
    #     # aggregate all the representations from `zs`
    #     cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
    #     att_outs = torch.cat((cls_tokens, att_outs), dim=1)  # (B, N+1, C)
    #     att_outs = att_outs + self.pos_embedding[:, :(n + 1)]  # (B, N+1, C)
    #     att_outs = self.dropout_attn(att_outs)  # (B, N+1, C)
    #
    #     att_outs = self.attn_layers(att_outs)  # (B, N+1, C)
    #     att_outs = self.norm_attn(att_outs)  # (B, N+1, C)
    #
    #     return att_outs[:, 0, :]  # attended cls_token: (B, C)

    def forward(self, x, one_sided_partial_mask=0., **kwargs):
        """
        :param x (B, C, L)
        """
        b = x.shape[0]
        n = len(self.blocks)
        out = x

        att_outs = torch.zeros(b, len(self.blocks), self.dim).to(x.device)  # (B, N, C); store representations from each block
        for i, block in enumerate(self.blocks):
            out, att_out = block(out)  # out:(B, C, L), att_out:(B, C)
            att_outs[:, i, :] = att_out

            if (i == 0) and (one_sided_partial_mask > 0.):
                # masking
                while True:
                    ind = np.random.rand(out.shape[-1]) > one_sided_partial_mask
                    if True in ind:
                        break

                out_elements = []
                for j, idx in enumerate(ind):
                    if idx == False:
                        out_elements.append(out[:, :, [j]])  # out[:, :, [j]]: (B, C, 1)
                    else:
                        mask_tokens = repeat(self.mask_token, '() c l -> b c l', b=b)
                        out_elements.append(mask_tokens)
                out = torch.cat(out_elements, dim=-1)  # (B, C, L)

        # aggregate all the representations from `zs`
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        att_outs = torch.cat((cls_tokens, att_outs), dim=1)  # (B, N+1, C)
        att_outs = att_outs + self.pos_embedding[:, :(n + 1)]  # (B, N+1, C)
        att_outs = self.dropout_attn(att_outs)  # (B, N+1, C)

        att_outs = self.attn_layers(att_outs)  # (B, N+1, C)
        # att_outs = self.norm_attn(att_outs)  # (B, N+1, C)
        att_out = att_outs[:, 0, :]  # attended cls_token: (B, C)

        # att_score = self.linear_attn(self.attn_layers(att_outs))  # (B, N, 1)
        # att_score = F.softmax(att_score, dim=1)  # (B, N, 1)
        # att_out = torch.sum(att_score * att_outs, dim=1)  # (B, C)

        # att_outs = self.attn_layers(att_outs)  # (B, N, C)
        # att_out = att_outs.mean(dim=1)  # (B, C)

        out = out.mean(dim=-1)  # (B, C)

        return out, att_out  # (B, C)


if __name__ == '__main__':
    x = torch.rand(1, 1, 100)  # (B, C, L)
    net = OnionNet(1,)
    out = net(x, one_sided_partial_mask=0.3)

    out.sum().backward()

    print(out)
    print(out.shape)