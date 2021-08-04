"""
# Define the Barlow Twins framework

# References
[1] J. Zbontar et al., 2021, "Barlow Twins: Self-Supervised Learning via Redundancy Reduction"
"""
import torch
import torch.nn as nn
from torch import relu

import matplotlib.pyplot as plt
import pandas as pd
import wandb

from vibcreg.backbone.resnet import ResNet1D, normalization_layer
from vibcreg.frameworks.framework_util_skeleton import Utility_SSL
from vibcreg.losses.barlow_twins_loss import barlow_twins_loss, barlow_twins_cross_correlation_mat


class Projector(nn.Module):
    def __init__(self, out_channels_backbone, proj_hid, proj_out, norm_layer_type_proj):
        super().__init__()
        # define layers
        self.linear1 = nn.Linear(out_channels_backbone, proj_hid)
        self.nl1 = normalization_layer(norm_layer_type_proj, proj_hid, dim=2)
        self.linear2 = nn.Linear(proj_hid, proj_hid)
        self.nl2 = normalization_layer(norm_layer_type_proj, proj_hid, dim=2)
        self.linear3 = nn.Linear(proj_hid, proj_out)

    def forward(self, x):
        out = relu(self.nl1(self.linear1(x)))
        out = relu(self.nl2(self.linear2(out)))
        out = self.linear3(out)
        return out


class BarlowTwins(nn.Module):
    def __init__(self, encoder: ResNet1D, last_channels_enc: int,
                 proj_hid_barlow: int = 4096, proj_out_barlow: int = 4096, norm_layer_type_proj_barlow: str = "BatchNorm", **kwargs):
        super().__init__()
        self.encoder = encoder
        self.projector = Projector(last_channels_enc, proj_hid_barlow, proj_out_barlow, norm_layer_type_proj_barlow)

    def forward(self, x1, x2):
        """
        :param x1: augmented view 1
        :param x2: augmented view 2
        """
        y1, y2 = self.encoder(x1), self.encoder(x2)  # (batch_size * feature_size)
        z1, z2 = self.projector(y1), self.projector(y2)
        return z1, z2


class Utility_BarlowTwins(Utility_SSL):
    def __init__(self, lambda_barlow=5e-3, **kwargs):
        super(Utility_BarlowTwins, self).__init__(**kwargs)
        self.lambda_barlow = lambda_barlow

    def wandb_watch(self, log_freq_watch=1000):
        if self.use_wandb:
            wandb.watch(self.rl_model.module.encoder, log_freq=log_freq_watch)
            wandb.watch(self.rl_model.module.projector, log_freq=log_freq_watch)

    @staticmethod
    def _batch_dim_wise_normalize_z(z):
        """batch dim.-wise normalization (standard-scaling style)"""
        mean = z.mean(dim=0)  # batch-wise mean
        std = z.std(dim=0)  # batch-wise std
        norm_z = (z - mean) / std  # standard-scaling; `dim=0`: batch dim.
        return norm_z

    def representation_learning(self, data_loader, optimizer, status):
        """
        :param status: train / validate / test
        """
        self.rl_model.train() if status == "train" else self.rl_model.eval()

        loss, step = 0., 0
        for subx_view1, subx_view2, label in data_loader:  # subx: (batch * n_channels * subseq_len)
            optimizer.zero_grad()

            z1, z2 = self.rl_model(subx_view1.to(self.device), subx_view2.to(self.device))

            # loss
            norm_z1, norm_z2 = self._batch_dim_wise_normalize_z(z1), self._batch_dim_wise_normalize_z(z2)  # standard-scaling norm.
            L = barlow_twins_loss(norm_z1, norm_z2, self.lambda_barlow)

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
            elif self.use_wandb and (status == "validate"):
                pass

        return loss / step

    @torch.no_grad()
    def log_cross_correlation_matrix(self, data_loader, n_features=40):
        if not self.use_wandb:
            return None

        self.rl_model.eval()

        count = 0
        partial_Cs = torch.zeros(n_features, n_features)
        for subx_view1, subx_view2, label in data_loader:  # subx: (batch * 12 * subseq_len); label: (batch * 71); 71 unique classes.
            subx_view1, subx_view2 = subx_view1.float(), subx_view2.float()
            z1, z2 = self.rl_model(subx_view1.to(self.device), subx_view2.to(self.device))
            norm_z1, norm_z2 = self._batch_dim_wise_normalize_z(z1), self._batch_dim_wise_normalize_z(z2)  # batch-wise standard-scaling norm.
            C = barlow_twins_cross_correlation_mat(norm_z1, norm_z2)
            partial_C = C[:n_features, :n_features]
            partial_Cs += partial_C.cpu()
            count += 1
        partial_Cs /= count
        partial_Cs = partial_Cs.abs()

        plt.figure(figsize=(10, 8))
        plt.gca().invert_yaxis()
        df = pd.DataFrame(partial_Cs.numpy())
        plt.pcolor(df, vmin=0., vmax=1.)
        plt.colorbar()
        plt.tight_layout()
        wandb.log({f'partial(abs(C))-ep_{self.epoch}': [wandb.Image(plt, caption=f'')]})  # caption=f'Partial cross-correlation matrix ({n_features}x{n_features})'
        plt.close()
