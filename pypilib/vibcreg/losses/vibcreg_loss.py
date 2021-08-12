
from torch import nn, Tensor
from .variance_loss import vibcreg_var_loss
from .covariance_loss import vibcreg_cov_loss
from .invariance_loss import vibcreg_invariance_loss


class VIbCRegLoss(nn.Module):
    def __init__(self, lambda_vibcreg: float = 25., mu_vibcreg: float = 25., nu_vibcreg: float = 200.):
        super(VIbCRegLoss, self).__init__()
        self._lambda = lambda_vibcreg
        self._mu = mu_vibcreg
        self._nu = nu_vibcreg

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        sim_loss = vibcreg_invariance_loss(z1, z2, self.loss_type_vibcreg)
        var_loss = vibcreg_var_loss(z1, z2)  # FcE loss
        cov_loss = vibcreg_cov_loss(z1, z2)
        return self._lambda * sim_loss + self._mu * var_loss + self._nu * cov_loss
