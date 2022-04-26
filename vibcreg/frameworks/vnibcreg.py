"""
VIbCReg: Variance Invariance better-Covariance Regularization

Reference:
[1] A. Bardes et al., 2021, "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import relu

from vibcreg.backbone.resnet import ResNet1D, normalization_layer
from vibcreg.frameworks.addin_msf import MemoryBank, AddinMSF
from vibcreg.frameworks.framework_util_skeleton import Utility_SSL
from vibcreg.losses.covariance_loss import vibcreg_cov_loss, vicreg_cov_loss
from vibcreg.losses.invariance_loss import vibcreg_invariance_loss
from vibcreg.losses.variance_loss import vibcreg_var_loss


class Projector(nn.Module):
    def __init__(self, last_channels_enc, proj_hid, proj_out, norm_layer_type_proj,
                 add_IterN_at_the_last_in_proj_vibcreg):
        super().__init__()
        self.add_IterN_at_the_last_in_proj_vibcreg = add_IterN_at_the_last_in_proj_vibcreg

        # define layers
        self.linear1 = nn.Linear(last_channels_enc, proj_hid)
        self.nl1 = normalization_layer(norm_layer_type_proj, proj_hid, dim=2)
        self.linear2 = nn.Linear(proj_hid, proj_hid)
        self.nl2 = normalization_layer(norm_layer_type_proj, proj_hid, dim=2)
        self.linear3 = nn.Linear(proj_hid, proj_out)
        self.nl3 = normalization_layer('IterNorm', proj_out,
                                       dim=2) if self.add_IterN_at_the_last_in_proj_vibcreg else None

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


class Discriminator(nn.Module):
    """
    TNC's discriminator
    """

    def __init__(self, input_size: int, w: float = 0.0):
        """
        :param input_size: size of the representation
        :param w:
        """
        super().__init__()
        self.input_size = input_size
        self.w = w

        self.model = nn.Sequential(nn.Linear(2 * self.input_size, 4 * self.input_size),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(4 * self.input_size, 1))
        nn.init.xavier_uniform_(self.model[0].weight)
        nn.init.xavier_uniform_(self.model[3].weight)

    def forward(self, x, x_tild):
        """
        Predict the probability of the two inputs belonging to the same neighbourhood.
        """
        x_all = torch.cat([x, x_tild], -1)
        out = self.model(x_all)
        return out


class VNIbCReg(nn.Module):
    def __init__(self, encoder: ResNet1D, last_channels_enc: int,
                 proj_hid_vibcreg: int = 4096, proj_out_vibcreg: int = 4096,
                 norm_layer_type_proj_vibcreg: str = "BatchNorm",
                 add_IterN_at_the_last_in_proj_vibcreg: bool = True,
                 use_predictor_msf: bool = False, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.use_predictor_msf = use_predictor_msf

        self.projector = Projector(last_channels_enc, proj_hid_vibcreg, proj_out_vibcreg, norm_layer_type_proj_vibcreg,
                                   add_IterN_at_the_last_in_proj_vibcreg)
        if self.use_predictor_msf:
            self.predictor = Predictor(proj_out_vibcreg)
        self.D = Discriminator(last_channels_enc)

    def forward(self, x1, x2, x3):
        """
        :param x1: augmented view 1 (reference)
        :param x2: augmented view 2 (neighboring)
        :param x3: augmented view 3 (non-neighboring)
        """
        # vibcreg
        y1, y2, y3 = self.encoder(x1), self.encoder(x2), self.encoder(x3)  # (batch_size * feature_size)
        z1, z2 = self.projector(y1), self.projector(y2)

        # tnc
        d_p = self.D(y1, y2)
        d_n = self.D(y1, y3)

        return (z1, z2), (y1, y2, y3), (d_p, d_n)


def tnc_loss(d_p, d_n,
             y, y_l, y_k,
             criterion,
             device,
             w: float):
    neighbors = torch.ones(d_p.shape).to(device)
    non_neighbors = torch.zeros(d_n.shape).to(device)

    p_loss = criterion(d_p, neighbors)
    n_loss = criterion(d_n, non_neighbors)
    n_loss_u = criterion(d_n, neighbors)
    loss = p_loss + (1 - w) * n_loss + w * n_loss_u  # original TNC loss

    p_acc = torch.sum(nn.Sigmoid()(d_p) > 0.5).item() / d_p.shape[0]
    n_acc = torch.sum(nn.Sigmoid()(d_n) < 0.5).item() / d_n.shape[0]

    # norm_y = F.normalize(y - y.mean(dim=1, keepdim=True), p=2, dim=1)  # (B, D)
    # norm_y_l = F.normalize(y_l - y_l.mean(dim=1, keepdim=True), p=2, dim=1)  # (B, D)
    # norm_y_k = F.normalize(y_k - y_k.mean(dim=1, keepdim=True), p=2, dim=1)  # (B, D)

    # corr_pos = torch.mm(norm_y, norm_y_l.T)  # (B, B)
    # corr_neg = torch.mm(norm_y, norm_y_k.T)  # (B, B)
    # diag_pos = torch.diag(corr_pos, 0)  # (B,)
    # diag_neg = torch.diag(corr_neg, 0)  # (B,)
    # diag_pos_loss = torch.mean((1. - diag_pos) ** 2)
    # diag_neg_loss = torch.mean((0. - diag_neg) ** 2)
    # if use_diag_loss:
    #     loss += (diag_pos_loss + diag_neg_loss)
    # loss += (diag_pos_loss + diag_neg_loss)

    # log
    loss_hist = {}
    loss_hist['TNC/tnc_loss'] = loss
    loss_hist['TNC/p_acc'] = p_acc
    loss_hist['TNC/n_acc'] = n_acc
    loss_hist['TNC/acc'] = (p_acc + n_acc) / 2
    # loss_hist['TNC/diag_pos_loss'] = diag_pos_loss
    # loss_hist['TNC/diag_neg_loss'] = diag_neg_loss

    return loss_hist


class Utility_VNIbCReg(Utility_SSL):
    def __init__(self, lambda_vibcreg=25, mu_vibcreg=25, nu_vibcreg=200, rho_tnc=25,
                 loss_type_vibcreg="mse", tnc_w: float = 0.05, use_vicreg_FD_loss=False, **kwargs):
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
        self.rho_tnc = rho_tnc
        self.loss_type_vibcreg = loss_type_vibcreg
        self.tnc_w = tnc_w
        self.use_vicreg_FD_loss = use_vicreg_FD_loss
        self.criterion_tnc = nn.BCEWithLogitsLoss()

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
        for subx_view1, subx_view2, subx_view3, label in data_loader:  # subx: (batch * n_channels * subseq_len)
            optimizer.zero_grad()

            (z1, z2), (y1, y2, y3), (d_p, d_n) = self.rl_model(subx_view1.to(self.device),
                                                               subx_view2.to(self.device),
                                                               subx_view3.to(self.device)
                                                               )

            # loss: vibcreg
            sim_loss = vibcreg_invariance_loss(z1, z2, self.loss_type_vibcreg)
            var_loss = vibcreg_var_loss(z1, z2)  # FcE loss
            cov_loss = vibcreg_cov_loss(z1, z2) if not self.use_vicreg_FD_loss else vicreg_cov_loss(z1, z2)  # FD loss
            L = self.lambda_vibcreg * sim_loss + self.mu_vibcreg * var_loss + self.nu_vibcreg * cov_loss

            # loss: tnc
            loss_hist = tnc_loss(d_p, d_n, y1, y2, y3, self.criterion_tnc, self.device, w=self.tnc_w)
            loss_hist['global_step'] = self.global_step
            L += self.rho_tnc * loss_hist['TNC/tnc_loss']

            # weight update
            if status == "train":
                L.backward()
                optimizer.step()
                self.lr_scheduler.step()
                self.global_step += 1

            loss += L.item()
            step += 1

            # status log
            loss_hist['loss'] = L
            loss_hist['sim_loss'] = sim_loss
            loss_hist['var_loss'] = var_loss
            loss_hist['cov_loss'] = cov_loss
            self.status_log_per_iter(status, y1, loss_hist)

        return loss / step

    def _representation_for_validation(self, x):
        return super()._representation_for_validation(x)
