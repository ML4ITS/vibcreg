"""
`invariance loss` == `similarity loss`.
"""
import torch
import torch.nn.functional as F
from torch import Tensor


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
