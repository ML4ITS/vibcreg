import numpy as np
import torch
from torch import Tensor, relu, nn
from torch.nn import functional as F


def vibcreg_cov_loss(z1: Tensor, z2: Tensor) -> Tensor:
    norm_z1 = (z1 - z1.mean(dim=0))
    norm_z2 = (z2 - z2.mean(dim=0))
    norm_z1 = F.normalize(norm_z1, p=2, dim=0)  # (batch * feature); l2-norm
    norm_z2 = F.normalize(norm_z2, p=2, dim=0)
    fxf_cov_z1 = torch.mm(norm_z1.T, norm_z1)  # (feature * feature)
    fxf_cov_z2 = torch.mm(norm_z2.T, norm_z2)
    ind = np.diag_indices(fxf_cov_z1.shape[0])
    dev = norm_z1.get_device()
    if dev >= 0:
        zer = torch.zeros(fxf_cov_z1.shape[0], device=dev)
    else:
        zer = torch.zeros(fxf_cov_z1.shape[0])
    fxf_cov_z1[ind[0], ind[1]] = zer
    fxf_cov_z2[ind[0], ind[1]] = zer
    cov_loss = (fxf_cov_z1 ** 2).mean() + (fxf_cov_z2 ** 2).mean()
    return cov_loss


def vicreg_cov_loss(z1: Tensor, z2: Tensor) -> Tensor:
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    N, D = z1.shape[0], z1.shape[1]  # batch_size, dimension size
    cov_z1 = torch.mm(z1.T, z1) / (N - 1)
    cov_z2 = torch.mm(z2.T, z2) / (N - 1)
    ind = np.diag_indices(cov_z1.shape[0])
    dev = z1.get_device()
    if dev >= 0:
        zer = torch.zeros(cov_z1.shape[0], device=dev)
    else:
        zer = torch.zeros(cov_z1.shape[0])
    cov_z1[ind[0], ind[1]] = zer
    cov_z2[ind[0], ind[1]] = zer
    cov_loss = (cov_z1 ** 2).sum() / D + (cov_z2 ** 2).sum() / D
    return cov_loss


def barlow_twins_cross_correlation_mat(norm_z1: Tensor, norm_z2: Tensor) -> Tensor:
    batch_size = norm_z1.shape[0]
    C = torch.mm(norm_z1.T, norm_z2) / batch_size
    return C


def barlow_twins_loss(norm_z1: Tensor, norm_z2: Tensor, lambda_: float):
    C = barlow_twins_cross_correlation_mat(norm_z1, norm_z2)

    # loss
    D = C.shape[0]
    identity_mat = torch.eye(D).to(C.get_device())
    C_diff = (identity_mat - C.to(C.get_device())) ** 2
    off_diagonal_mul = (lambda_ * torch.abs(identity_mat - 1)) + identity_mat
    loss = (C_diff * off_diagonal_mul).sum()
    return loss


mse_loss = torch.nn.MSELoss()


def cos_sim_loss(z1: Tensor, z2: Tensor) -> Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return 1 - (z1 * z2).sum(dim=1).mean()


def vibcreg_invariance_loss(z1: Tensor, z2: Tensor, loss_type_vibcreg: str) -> Tensor:
    sim_loss = 0.
    if loss_type_vibcreg == 'mse':
        sim_loss = mse_loss(z1, z2)
    elif loss_type_vibcreg == 'cos_sim':
        sim_loss = cos_sim_loss(z1, z2)
    elif loss_type_vibcreg == 'hybrid':
        sim_loss = 0.5 * mse_loss(z1, z2) + 0.5 * cos_sim_loss(z1, z2)
    return sim_loss


def simsiam_cos_sim_loss(p, z):
    """
    :param p: output from `predictor`
    :param z: output from `projector`

    SimSiam's cosine similarity loss.
    """
    z.detach()  # stop gradient

    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return 1 - (p * z).sum(dim=1).mean()


def vibcreg_var_loss(z1: Tensor, z2: Tensor) -> Tensor:
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    FcE_scale = 1.
    var_loss = torch.mean(relu(FcE_scale - std_z1)) + torch.mean(relu(FcE_scale - std_z2))
    return var_loss


class VIbCRegLoss(nn.Module):
    def __init__(self, lambda_vibcreg: float = 25., mu_vibcreg: float = 25., nu_vibcreg: float = 200.,
                 loss_type_vibcreg: str = "mse"):
        super(VIbCRegLoss, self).__init__()
        self._lambda = lambda_vibcreg
        self._mu = mu_vibcreg
        self._nu = nu_vibcreg
        self.loss_type_vibcreg = loss_type_vibcreg

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        sim_loss = vibcreg_invariance_loss(z1, z2, self.loss_type_vibcreg)
        var_loss = vibcreg_var_loss(z1, z2)  # FcE loss
        cov_loss = vibcreg_cov_loss(z1, z2)
        return self._lambda * sim_loss + self._mu * var_loss + self._nu * cov_loss
