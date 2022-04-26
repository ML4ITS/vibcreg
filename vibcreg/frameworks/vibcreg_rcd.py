"""
VIbCReg: Variance Invariance better-Covariance Regularization

Reference:
[1] A. Bardes et al., 2021, "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"
"""
import torch
import torch.nn as nn
import wandb
from torch import relu

from vibcreg.backbone.resnet import ResNet1D, normalization_layer
from vibcreg.frameworks.addin_msf import MemoryBank, AddinMSF
from vibcreg.frameworks.framework_util_skeleton import Utility_SSL
from vibcreg.losses.covariance_loss import vibcreg_cov_loss, vicreg_cov_loss
from vibcreg.losses.invariance_loss import vibcreg_invariance_loss
from vibcreg.losses.variance_loss import vibcreg_var_loss


class Projector(nn.Module):
    def __init__(self, last_channels_enc, proj_hid, proj_out, norm_layer_type_proj, add_IterN_at_the_last_in_proj_vibcreg):
        super().__init__()
        self.add_IterN_at_the_last_in_proj_vibcreg = add_IterN_at_the_last_in_proj_vibcreg

        # define layers
        self.linear1 = nn.Linear(last_channels_enc, proj_hid)
        self.nl1 = normalization_layer(norm_layer_type_proj, proj_hid, dim=2)
        self.linear2 = nn.Linear(proj_hid, proj_hid)
        self.nl2 = normalization_layer(norm_layer_type_proj, proj_hid, dim=2)
        self.linear3 = nn.Linear(proj_hid, proj_out)
        self.nl3 = normalization_layer('IterNorm', proj_out, dim=2) if self.add_IterN_at_the_last_in_proj_vibcreg else None

    def forward(self, x):
        out = relu(self.nl1(self.linear1(x)))
        out = relu(self.nl2(self.linear2(out)))
        out = self.linear3(out)  # (batch x feature)
        if self.add_IterN_at_the_last_in_proj_vibcreg:
            out = self.nl3(out)

        return out


class Predictor(nn.Module):
    def __init__(self, in_size, h_size, out_size):
        super(Predictor, self).__init__()
        # define layers
        self.linear1 = nn.Linear(in_size, h_size)
        self.nl1 = nn.BatchNorm1d(h_size)
        self.linear2 = nn.Linear(h_size, out_size)

    def forward(self, x):
        out = relu(self.nl1(self.linear1(x)))
        out = torch.sigmoid(self.linear2(out))
        return out


class VIbCRegRCD(nn.Module):
    def __init__(self, encoder: ResNet1D, last_channels_enc: int,
                 proj_hid_vibcreg: int = 4096, proj_out_vibcreg: int = 4096, norm_layer_type_proj_vibcreg: str = "BatchNorm", add_IterN_at_the_last_in_proj_vibcreg: bool = True,
                 **kwargs):
        super().__init__()
        self.encoder = encoder
        self.projector = Projector(last_channels_enc, proj_hid_vibcreg, proj_out_vibcreg, norm_layer_type_proj_vibcreg, add_IterN_at_the_last_in_proj_vibcreg)
        self.predictor = Predictor(2*last_channels_enc, last_channels_enc, 1)

    def forward(self, x1, x2):
        """
        :param x1: augmented view 1
        :param x2: augmented view 2
        """
        # vibcreg
        y1, y2 = self.encoder(x1), self.encoder(x2)  # (batch_size * feature_size)
        z1, z2 = self.projector(y1), self.projector(y2)

        # pretext task
        ys = torch.cat((y1, y2), dim=1)
        rcd_hat = self.predictor(ys)

        return y1, y2, z1, z2, rcd_hat


class Utility_VIbCRegRCD(Utility_SSL):
    def __init__(self, lambda_vibcreg=25, mu_vibcreg=25, nu_vibcreg=200,
                 loss_type_vibcreg="mse", use_vicreg_FD_loss=False, **kwargs):
        """
        :param lambda_vibcreg: weight for the similarity (invariance) loss.
        :param mu_vibcreg: weight for the FcE (variance) loss
        :param nu_vibcreg: weight for the FD (covariance) loss
        :param loss_type_vibcreg:
        :param use_vicreg_FD_loss: if True, VICReg's FD loss is used instead of VIbCReg's.
        :param kwargs: params belonging to the `Utility_SSL` class.
        """
        super().__init__(**kwargs)
        self.lambda_vibcreg = lambda_vibcreg
        self.mu_vibcreg = mu_vibcreg
        self.nu_vibcreg = nu_vibcreg
        self.loss_type_vibcreg = loss_type_vibcreg
        self.use_vicreg_FD_loss = use_vicreg_FD_loss
        self.l2_loss = nn.MSELoss()

    def wandb_watch(self):
        if self.use_wandb:
            wandb.watch(self.rl_model.module.encoder)
            wandb.watch(self.rl_model.module.projector)

    def status_log_per_iter(self, status, z, loss_hist: dict):
        loss_hist = {k + f'.{status}': v for k, v in loss_hist.items()}
        loss_hist['global_step'] = self.global_step
        wandb.log(
            {'global_step': self.global_step, 'feature_comp_expr_metrics': self._feature_comp_expressiveness_metrics(z),
             'feature_decorr_metrics': self._compute_feature_decorr_metrics(z)})
        wandb.log(loss_hist)

    def representation_learning(self, data_loader, optimizer, status):
        """
        :param status: train / validate / test
        """
        self.rl_model.train() if status == "train" else self.rl_model.eval()

        loss, step = 0., 0
        for batch in data_loader:  # subx: (batch * n_channels * subseq_len)
            subx_view1, subx_view2, label, rc_dist = batch
            optimizer.zero_grad()

            y1, y2, z1, z2, rcd_hat = self.rl_model(subx_view1.to(self.device), subx_view2.to(self.device))

            # loss: vibcreg
            sim_loss = vibcreg_invariance_loss(z1, z2, self.loss_type_vibcreg)
            var_loss = vibcreg_var_loss(z1, z2)  # FcE loss
            cov_loss = vibcreg_cov_loss(z1, z2) if not self.use_vicreg_FD_loss else vicreg_cov_loss(z1, z2)  # FD loss
            L = self.lambda_vibcreg * sim_loss + self.mu_vibcreg * var_loss + self.nu_vibcreg * cov_loss

            # loss: pretext task
            rcd_loss = self.l2_loss(rc_dist.float().to(self.device), rcd_hat)
            L += self.lambda_vibcreg * rcd_loss

            # weight update
            if status == "train":
                L.backward()
                optimizer.step()
                self.lr_scheduler.step()
                self.global_step += 1

            loss += L.item()
            step += 1

            # status log
            loss_hist = {}
            loss_hist['loss'] = L
            loss_hist['sim_loss'] = sim_loss
            loss_hist['var_loss'] = var_loss
            loss_hist['cov_loss'] = cov_loss
            loss_hist['rcd_loss'] = rcd_loss
            self.status_log_per_iter(status, y1, loss_hist)

        if step == 0:
            return 0.
        else:
            return loss / step

    def _representation_for_validation(self, x):
        return super()._representation_for_validation(x)

