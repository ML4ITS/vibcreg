"""
SimCLR

Reference:
[1] https://github.com/mdiephuis/SimCLR
"""
import torch
import torch.nn as nn
import wandb

from vibcreg.backbone.resnet import ResNet1D
from vibcreg.frameworks.framework_util_skeleton import Utility_SSL


class Projector(nn.Module):
    def __init__(self, last_channels_enc, proj_hid, proj_out):
        super().__init__()

        self.proj_head = nn.Sequential(
            nn.Linear(last_channels_enc, proj_hid),
            nn.BatchNorm1d(proj_hid),
            nn.ReLU(),
            nn.Linear(proj_hid, proj_out)
        )

    def forward(self, x):
        out = self.proj_head(x)
        return out


class SimCLR(nn.Module):
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
        return z1, z2


class contrastive_loss(nn.Module):
    def __init__(self, tau : float, normalize : bool):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, xi, xj, device):

        x = torch.cat((xi, xj), dim=0)

        # is_cuda = x.is_cuda
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
        norm_sum = norm_sum.to(device) #if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss


class Utility_SimCLR(Utility_SSL):
    def __init__(self, tau: float, normalize: bool, **kwargs):
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

            z1, z2 = self.rl_model(subx_view1.to(self.device), subx_view2.to(self.device))

            L = self.loss_func(z1, z2, self.device)

            # weight update
            if status == "train":
                L.backward()
                optimizer.step()
                self.lr_scheduler.step()
                self.global_step += 1

            loss += L.item()
            step += 1

            # status log
            loss_hist = {'loss': L}
            self.status_log_per_iter(status, z1, loss_hist)

        if step == 0:
            return 0.
        else:
            return loss / step

    def _representation_for_validation(self, x):
        return super()._representation_for_validation(x)

