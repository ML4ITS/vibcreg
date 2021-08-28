from typing import Callable, Tuple
from torch import nn, relu, Tensor
from .iter_norm import IterNorm


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


class Projector(nn.Module):
    def __init__(self, last_channels_enc, proj_hid, proj_out, norm_layer_type_proj):
        super().__init__()

        # define layers
        self.n_channels_enc = last_channels_enc
        self.n_out_dims = proj_out
        self.linear1 = nn.Linear(last_channels_enc, proj_hid)
        self.nl1 = normalization_layer(norm_layer_type_proj, proj_hid, dim=2)
        self.linear2 = nn.Linear(proj_hid, proj_hid)
        self.nl2 = normalization_layer(norm_layer_type_proj, proj_hid, dim=2)
        self.linear3 = nn.Linear(proj_hid, proj_out)
        self.nl3 = normalization_layer('IterNorm', proj_out, dim=2)

    def forward(self, x: Tensor) -> Tensor:
        out = relu(self.nl1(self.linear1(x)))
        out = relu(self.nl2(self.linear2(out)))
        out = self.linear3(out)  # (batch x feature)
        out = self.nl3(out)
        return out


class VIbCReg(nn.Module):
    def __init__(self, encoder: Callable, last_channels_enc: int,
                 proj_hid_vibcreg: int = 4096,
                 proj_out_vibcreg: int = 4096,
                 norm_layer_type_proj_vibcreg: str = "BatchNorm"):
        super().__init__()
        self.encoder_out_dim = last_channels_enc
        self.encoder = encoder()

        self.projector = Projector(last_channels_enc, proj_hid_vibcreg, proj_out_vibcreg, norm_layer_type_proj_vibcreg)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x1: augmented view 1
        :param x2: augmented view 2
        """
        y = self.encoder(x)
        z = self.projector(y)
        return z
        # y1, y2 = self.encoder(x1), self.encoder(x2)  # (batch_size * feature_size)
        # z1, z2 = self.projector(y1), self.projector(y2)
        # return z1, z2
