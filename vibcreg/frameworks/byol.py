"""
BYOL

Reference:
[1] https://github.com/sthalles/PyTorch-BYOL
"""
import copy
import torch
import torch.nn as nn
import wandb

import torch.nn.functional as F
from vibcreg.backbone.resnet import ResNet1D
from vibcreg.frameworks.framework_util_skeleton import Utility_SSL


# class MLPHead(nn.Module):
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


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    def __init__(self, encoder: ResNet1D, last_channels_enc: int,
                 proj_hid_vibcreg: int = 512, proj_out_vibcreg: int = 512,
                 momentum: float = 0.9,
                 **kwargs):
        super().__init__()
        self.m = momentum

        self.encoder = encoder
        self.projector = MLPHead(last_channels_enc, proj_hid_vibcreg, proj_out_vibcreg)
        self.online_network = nn.Sequential(self.encoder, self.projector)

        self.predictor = MLPHead(last_channels_enc, proj_hid_vibcreg, proj_out_vibcreg)

        self.target_network = copy.deepcopy(self.online_network)
        self.initializes_target_network()

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def forward(self, x1, x2, status: str):
        """
        :param x1: augmented view 1
        :param x2: augmented view 2
        """
        # compute query feature
        z1 = self.online_network(x1)
        z2 = self.online_network(x2)
        predictions_from_view_1 = self.predictor(z1)
        predictions_from_view_2 = self.predictor(z2)

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(x1)
            targets_to_view_1 = self.target_network(x2)

        # update the target network (the key encoder)
        if status == 'train':
            self._update_target_network_parameters()

        return (z1, z2), (predictions_from_view_1, predictions_from_view_2), (targets_to_view_1, targets_to_view_2)


class Utility_BYOL(Utility_SSL):
    def __init__(self, **kwargs):
        """
        :param lambda_vibcreg: weight for the similarity (invariance) loss.
        :param mu_vibcreg: weight for the FcE (variance) loss
        :param nu_vibcreg: weight for the FD (covariance) loss
        :param loss_type_vibcreg:
        :param use_vicreg_FD_loss: if True, VICReg's FD loss is used instead of VIbCReg's.
        :param kwargs: params belonging to the `Utility_SSL` class.
        """
        super().__init__(**kwargs)

    def wandb_watch(self):
        if self.use_wandb:
            wandb.watch(self.rl_model.module.encoder)
            # wandb.watch(self.rl_model.module.projector)

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

            # compute query feature
            (z1, z2), (predictions_from_view_1, predictions_from_view_2), (targets_to_view_1, targets_to_view_2) = self.rl_model(subx_view1.to(self.device), subx_view2.to(self.device), status)

            # compute key features
            L = self.rl_model.module.regression_loss(predictions_from_view_1, targets_to_view_1)
            L += self.rl_model.module.regression_loss(predictions_from_view_2, targets_to_view_2)
            L = L.mean()

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

