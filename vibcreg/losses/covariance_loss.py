import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def vibcreg_cov_loss(z1: Tensor, z2: Tensor) -> Tensor:
    norm_z1 = (z1 - z1.mean(dim=0))
    norm_z2 = (z2 - z2.mean(dim=0))
    norm_z1 = F.normalize(norm_z1, p=2, dim=0)  # (batch * feature); l2-norm
    norm_z2 = F.normalize(norm_z2, p=2, dim=0)
    fxf_cov_z1 = torch.mm(norm_z1.T, norm_z1)  # (feature * feature)
    fxf_cov_z2 = torch.mm(norm_z2.T, norm_z2)
    ind = np.diag_indices(fxf_cov_z1.shape[0])
    fxf_cov_z1[ind[0], ind[1]] = torch.zeros(fxf_cov_z1.shape[0]).to(norm_z1.get_device())
    fxf_cov_z2[ind[0], ind[1]] = torch.zeros(fxf_cov_z2.shape[0]).to(norm_z1.get_device())
    cov_loss = (fxf_cov_z1 ** 2).mean() + (fxf_cov_z2 ** 2).mean()
    return cov_loss


def vicreg_cov_loss(z1: Tensor, z2: Tensor) -> Tensor:
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    N, D = z1.shape[0], z1.shape[1]  # batch_size, dimension size
    cov_z1 = torch.mm(z1.T, z1) / (N - 1)
    cov_z2 = torch.mm(z2.T, z2) / (N - 1)
    ind = np.diag_indices(cov_z1.shape[0])
    cov_z1[ind[0], ind[1]] = torch.zeros(cov_z1.shape[0], device=z1.get_device())  # off-diagonal(..)
    cov_z2[ind[0], ind[1]] = torch.zeros(cov_z2.shape[0], device=z1.get_device())  # off-diagonal(..)
    cov_loss = (cov_z1 ** 2).sum() / D + (cov_z2 ** 2).sum() / D
    return cov_loss
