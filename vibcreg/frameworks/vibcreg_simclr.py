"""
SimCLR

Reference:
[1] https://github.com/mdiephuis/SimCLR
"""
import torch
import torch.nn as nn
import wandb

from vibcreg.backbone.resnet import ResNet1D, normalization_layer
from vibcreg.backbone.resnet import ResNet1D
from vibcreg.frameworks.framework_util_skeleton import Utility_SSL
from vibcreg.losses.covariance_loss import vibcreg_cov_loss, vicreg_cov_loss
from vibcreg.losses.invariance_loss import vibcreg_invariance_loss
from vibcreg.losses.variance_loss import vibcreg_var_loss


# class Projector(nn.Module):
#     def __init__(self, last_channels_enc, proj_hid, proj_out):
#         super().__init__()
#
#         self.proj_head = nn.Sequential(
#             nn.Linear(last_channels_enc, proj_hid),
#             nn.BatchNorm1d(proj_hid),
#             nn.ReLU(),
#             nn.Linear(proj_hid, proj_out)
#         )
#
#     def forward(self, x):
#         out = self.proj_head(x)
#         return out


class Projector(nn.Module):
    def __init__(self,
                 last_channels_enc,
                 proj_hid,
                 proj_out,
                 norm_layer_type_proj="BatchNorm",
                 add_IterN_at_the_last_in_proj_vibcreg=True):
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
        out = torch.relu(self.nl1(self.linear1(x)))
        out = torch.relu(self.nl2(self.linear2(out)))
        out = self.linear3(out)  # (batch x feature)
        if self.add_IterN_at_the_last_in_proj_vibcreg:
            out = self.nl3(out)

        return out

class VIbCRegSimCLR(nn.Module):
    def __init__(self, encoder: ResNet1D, last_channels_enc: int,
                 proj_hid_vibcreg: int = 4096, proj_out_vibcreg: int = 4096, **kwargs):
        super().__init__()
        self.encoder = encoder

        self.projector = Projector(last_channels_enc, proj_hid_vibcreg, proj_out_vibcreg)

    def forward(self, x1, x2):
        """
        :param x1: augmented view 1
        :param x2: augmented view 2
        """
        y1, y2 = self.encoder(x1), self.encoder(x2)  # (batch_size * feature_size)
        z1, z2 = self.projector(y1), self.projector(y2)
        return y1, y2, z1, z2


class contrastive_loss(nn.Module):
    def __init__(self, tau : float, normalize : bool):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, xi, xj):

        x = torch.cat((xi, xj), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.T)
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss


class Utility_VIbCRegSimCLR(Utility_SSL):
    def __init__(self,
                 tau: float, normalize: bool,
                 lambda_vibcreg=25, mu_vibcreg=25, nu_vibcreg=200,
                 **kwargs):
        """
        :param lambda_vibcreg: weight for the similarity (invariance) loss.
        :param mu_vibcreg: weight for the FcE (variance) loss
        :param nu_vibcreg: weight for the FD (covariance) loss
        :param loss_type_vibcreg:
        :param use_vicreg_FD_loss: if True, VICReg's FD loss is used instead of VIbCReg's.
        :param kwargs: params belonging to the `Utility_SSL` class.
        """
        super().__init__(**kwargs)
        self.loss_func = contrastive_loss(tau=tau, normalize=normalize)
        self.lambda_vibcreg = lambda_vibcreg
        self.mu_vibcreg = mu_vibcreg
        self.nu_vibcreg = nu_vibcreg

    def wandb_watch(self):
        if self.use_wandb:
            wandb.watch(self.rl_model.module.encoder)
            wandb.watch(self.rl_model.module.projector)

    def status_log_per_iter(self, status, z, loss_hist: dict):
        loss_hist = {k + f'.{status}': v for k, v in loss_hist.items()}
        loss_hist['global_step'] = self.global_step
        wandb.log({'global_step': self.global_step, 'feature_comp_expr_metrics': self._feature_comp_expressiveness_metrics(z), 'feature_decorr_metrics': self._compute_feature_decorr_metrics(z)})
        wandb.log(loss_hist)

    def representation_learning(self, data_loader, optimizer, status):
        """
        :param status: train / validate / test
        """
        self.rl_model.train() if status == "train" else self.rl_model.eval()

        loss, step = 0., 0
        for subx_view1, subx_view2, label in data_loader:  # subx: (batch * n_channels * subseq_len)
            optimizer.zero_grad()

            y1, y2, z1, z2 = self.rl_model(subx_view1.to(self.device), subx_view2.to(self.device))

            # loss: vibcreg
            sim_loss = vibcreg_invariance_loss(z1, z2, 'mse')
            var_loss = vibcreg_var_loss(z1, z2)  # FcE loss
            cov_loss = vibcreg_cov_loss(z1, z2)
            L = self.lambda_vibcreg * sim_loss + self.mu_vibcreg * var_loss + self.nu_vibcreg * cov_loss

            # loss: simclr
            simclr_loss = self.loss_func(z1, z2)
            L += simclr_loss

            # weight update
            if status == "train":
                L.backward()
                optimizer.step()
                self.lr_scheduler.step()
                self.global_step += 1

            loss += L.item()
            step += 1

            # status log
            loss_hist = {'loss': L,
                         'sim_loss': sim_loss,
                         'var_loss': var_loss,
                         'cov_loss': cov_loss,
                         'simclr_loss': simclr_loss,
                         }
            self.status_log_per_iter(status, z1, loss_hist)

        if step == 0:
            return 0.
        else:
            return loss / step

    def _representation_for_validation(self, x):
        return super()._representation_for_validation(x)

