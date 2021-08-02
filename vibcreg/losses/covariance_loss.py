import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F


class VIbCRegCovLoss(nn.Module):

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        norm_z1 = (z1 - z1.mean(dim=0))
        norm_z2 = (z2 - z2.mean(dim=0))
        norm_z1 = F.normalize(norm_z1, p=2, dim=0)  # (batch * feature); l2-norm
        norm_z2 = F.normalize(norm_z2, p=2, dim=0)
        fxf_cov_z1 = torch.mm(norm_z1.T, norm_z1)  # (feature * feature)
        fxf_cov_z2 = torch.mm(norm_z2.T, norm_z2)
        ind = np.diag_indices(fxf_cov_z1.shape[0])
        fxf_cov_z1[ind[0], ind[1]] = torch.zeros(fxf_cov_z1.shape[0])
        fxf_cov_z2[ind[0], ind[1]] = torch.zeros(fxf_cov_z2.shape[0])
        cov_loss = (fxf_cov_z1 ** 2).mean() + (fxf_cov_z2 ** 2).mean()
        return cov_loss


class VICRegDecorrLoss(nn.Module):

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        norm_z1 = (z1 - z1.mean(dim=0))
        norm_z2 = (z2 - z2.mean(dim=0))
        norm_z1 = F.normalize(norm_z1, p=2, dim=0)  # (batch * feature); l2-norm
        norm_z2 = F.normalize(norm_z2, p=2, dim=0)
        fxf_cov_z1 = torch.mm(norm_z1.T, norm_z1)  # (feature * feature)
        fxf_cov_z2 = torch.mm(norm_z2.T, norm_z2)
        ind = np.diag_indices(fxf_cov_z1.shape[0])
        fxf_cov_z1[ind[0], ind[1]] = torch.zeros(fxf_cov_z1.shape[0]).to(self.device)
        fxf_cov_z2[ind[0], ind[1]] = torch.zeros(fxf_cov_z2.shape[0]).to(self.device)
        cov_loss = (fxf_cov_z1 ** 2).mean() + (fxf_cov_z2 ** 2).mean()
        return cov_loss
