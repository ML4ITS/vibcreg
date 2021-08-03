"""
VIbCReg: Variance Invariance better-Covariance Regularization

Reference:
[1] A. Bardes et al., 2021, "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import relu

import wandb
import numpy as np

from vibcreg.backbone.resnet import ResNet1D, normalization_layer
from vibcreg.frameworks.framework_util_skeleton import Utility_SSL
from vibcreg.frameworks.addin_msf import MemoryBank, AddinMSF

from vibcreg.losses.covariance_loss import vibcreg_cov_loss, vicreg_cov_loss
from vibcreg.losses.variance_loss import vibcreg_var_loss
from vibcreg.losses.invariance_loss import vibcreg_invariance_loss


class Projector(nn.Module):
    def __init__(self, out_channels_backbone, proj_hid, proj_out, norm_layer_type_proj, add_IterN_at_the_last_in_proj_vibcreg):
        super().__init__()
        self.add_IterN_at_the_last_in_proj_vibcreg = add_IterN_at_the_last_in_proj_vibcreg

        # define layers
        self.linear1 = nn.Linear(out_channels_backbone, proj_hid)
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
    """
    predictor from the MSF paper.
    """
    def __init__(self, proj_out_vibcreg):
        super(Predictor, self).__init__()
        # define layers
        self.linear1 = nn.Linear(proj_out_vibcreg, proj_out_vibcreg)
        self.nl1 = nn.BatchNorm1d(proj_out_vibcreg)
        self.linear2 = nn.Linear(proj_out_vibcreg, proj_out_vibcreg)

    def forward(self, x):
        out = relu(self.nl1(self.linear1(x)))
        out = self.linear2(out)
        return out


class VIbCReg(nn.Module):
    def __init__(self, encoder: ResNet1D, out_channels_backbone: int,
                 proj_hid_vibcreg: int = 4096, proj_out_vibcreg: int = 4096, norm_layer_type_proj_vibcreg: str = "BatchNorm", add_IterN_at_the_last_in_proj_vibcreg: bool = True,
                 weight_on_msfLoss: float = 0., use_predictor_msf: bool = False, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.weight_on_msfLoss = weight_on_msfLoss
        self.use_predictor_msf = use_predictor_msf

        self.projector = Projector(out_channels_backbone, proj_hid_vibcreg, proj_out_vibcreg, norm_layer_type_proj_vibcreg, add_IterN_at_the_last_in_proj_vibcreg)
        if self.weight_on_msfLoss and self.use_predictor_msf:
            self.predictor = Predictor(proj_out_vibcreg)

    def forward(self, x1, x2):
        """
        :param x1: augmented view 1
        :param x2: augmented view 2
        """
        y1, y2 = self.encoder(x1), self.encoder(x2)  # (batch_size * feature_size)
        z1, z2 = self.projector(y1), self.projector(y2)
        return z1, z2


class Utility_VIbCReg(Utility_SSL):
    def __init__(self, lambda_vibcreg=25, mu_vibcreg=25, nu_vibcreg=200, loss_type_vibcreg="mse", use_vicreg_FD_loss=False, **kwargs):
        """
        :param lambda_vibcreg: weight for the similarity (invariance) loss.
        :param mu_vibcreg: weight for the FcE (variance) loss
        :param nu_vibcreg: weight for the FD (covariance) loss
        :param loss_type_vibcreg:
        :param use_vicreg_FD_loss: if True, VICReg's FD loss is used instead of VIbCReg's.
        :param kwargs: params belonging to the `Utility_SSL` class.
        """
        super(Utility_VIbCReg, self).__init__(**kwargs)
        self.lambda_vibcreg = lambda_vibcreg
        self.mu_vibcreg = mu_vibcreg
        self.nu_vibcreg = nu_vibcreg
        self.loss_type_vibcreg = loss_type_vibcreg
        self.use_vicreg_FD_loss = use_vicreg_FD_loss

        self.amsf = None
        self.use_predictor_msf = None

    def create_msf(self, size_mb, k_msf, device_ids, feature_size, batch_size, tau_target_net=0.99, use_EMAN=False, use_predictor_msf=False):
        memory_bank = MemoryBank(size_mb, k_msf, device_ids, feature_size, batch_size)
        self.amsf = AddinMSF(memory_bank, tau_target_net, use_EMAN)
        self.amsf.create_target_net(self.rl_model)
        self.use_predictor_msf = use_predictor_msf

    def wandb_watch(self, log_freq_watch=1000):
        if self.use_wandb:
            wandb.watch(self.rl_model.module.encoder, log_freq=log_freq_watch)
            wandb.watch(self.rl_model.module.projector, log_freq=log_freq_watch)
            if self.weight_on_msfLoss and self.use_predictor_msf:
                wandb.watch(self.rl_model.module.predictor, log_freq=log_freq_watch)

    def representation_learning(self, data_loader, optimizer, status):
        """
        :param status: train / validate / test
        """
        self.rl_model.train() if status == "train" else self.rl_model.eval()

        loss, step = 0., 0
        for subx_view1, subx_view2, label in data_loader:  # subx: (batch * n_channels * subseq_len)
            optimizer.zero_grad()

            z1, z2 = self.rl_model(subx_view1.to(self.device), subx_view2.to(self.device))

            sim_loss = vibcreg_invariance_loss(z1, z2, self.loss_type_vibcreg)
            var_loss = vibcreg_var_loss(z1, z2)  # FcE loss
            cov_loss = vibcreg_cov_loss(z1, z2) if not self.use_vicreg_FD_loss else vicreg_cov_loss(z1, z2)  # FD loss
            L = self.lambda_vibcreg * sim_loss + self.mu_vibcreg * var_loss + self.nu_vibcreg * cov_loss

            if self.weight_on_msfLoss:
                self.amsf.update_target_net(self.rl_model)
                z1_target, _ = self.amsf.target_net(subx_view1.to(self.device), subx_view2.to(self.device))
                if self.weight_on_msfLoss and self.use_predictor_msf:
                    z2 = self.rl_model.module.predictor(z2)
                msf_loss = self.amsf.compute_loss(z_target=z1_target, z_online=z2)
                L += self.weight_on_msfLoss * msf_loss

            # weight update
            if status == "train":
                L.backward()
                optimizer.step()
                self.lr_scheduler.step()

            loss += L.item()
            step += 1

            # status log
            if self.use_wandb and (status == "train"):
                self.global_step += 1
                wandb.log({'global_step': self.global_step, 'feature_comp_expr_metrics': self._feature_comp_expressiveness_metrics(z1), 'feature_decorr_metrics': self._compute_feature_decorr_metrics(z1)})
                wandb.log({'global_step': self.global_step, 'sim_loss': sim_loss, 'var_loss': var_loss, 'cov_loss': cov_loss})
                wandb.log({'global_step': self.global_step, 'msf_loss': msf_loss}) if self.weight_on_msfLoss else None
            elif self.use_wandb and (status == "validate"):
                wandb.log({'global_step': self.global_step, 'sim_loss_val': sim_loss, 'var_loss_val': var_loss, 'cov_loss_val': cov_loss})
                wandb.log({'global_step': self.global_step, 'msf_loss_val': msf_loss}) if self.weight_on_msfLoss else None

        # compute kNN accuracy during validation
        if self.use_wandb and (status == "validate"):
            self.log_kNN_acc_during_validation(data_loader)  # `val_data_loader` is fed.

        return loss / step

