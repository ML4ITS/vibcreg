import torch
from torch import Tensor


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
