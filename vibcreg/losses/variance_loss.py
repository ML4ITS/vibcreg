import torch
from torch import Tensor, relu


def vibcreg_var_loss(z1: Tensor, z2: Tensor) -> Tensor:
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    FcE_scale = 1.
    var_loss = torch.mean(relu(FcE_scale - std_z1)) + torch.mean(relu(FcE_scale - std_z2))
    return var_loss
