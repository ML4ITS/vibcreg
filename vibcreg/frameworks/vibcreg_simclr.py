"""
SimCLR

Reference:
[1] https://github.com/mdiephuis/SimCLR
"""
import numpy as np
import torch
import torch.nn as nn
import wandb

from vibcreg.backbone.resnet import ResNet1D, normalization_layer
from vibcreg.backbone.resnet import ResNet1D
from vibcreg.frameworks.framework_util_skeleton import Utility_SSL
from vibcreg.losses.covariance_loss import vibcreg_cov_loss, vicreg_cov_loss
from vibcreg.losses.invariance_loss import vibcreg_invariance_loss
from vibcreg.losses.variance_loss import vibcreg_var_loss


class ProjectorSimCLR(nn.Module):
    def __init__(self, last_channels_enc, proj_hid, proj_out):
        super().__init__()

        self.proj_head = nn.Sequential(
            nn.Linear(last_channels_enc, proj_hid),
            nn.BatchNorm1d(proj_hid),
            nn.ReLU(),
            nn.Linear(proj_hid, proj_hid),
            nn.BatchNorm1d(proj_hid),
            nn.ReLU(),
            nn.Linear(proj_hid, proj_out),
            # normalization_layer('IterNorm', proj_out, dim=2)
        )

    def forward(self, x):
        out = self.proj_head(x)
        return out


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
                 proj_hid_vibcreg: int = 4096, proj_out_vibcreg: int = 4096,
                 **kwargs):
        super().__init__()
        self.encoder = encoder

        self.pool_type = kwargs.get('pool_type', None)
        # last_channels_enc = 3 * last_channels_enc if self.pool_type == 'combined' else last_channels_enc
        self.projector = Projector(last_channels_enc, proj_hid_vibcreg, proj_out_vibcreg)
        self.projector_simclr = ProjectorSimCLR(last_channels_enc, proj_hid_vibcreg, proj_out_vibcreg)

    def forward(self, x1, x2, projector_type: str = 'vibcreg', return_y=False):
        """
        :param x1: augmented view 1
        :param x2: augmented view 2
        """
        # if projector_type == 'vibcreg':
        #     y1, y2 = self.encoder(x1), self.encoder(x2)  # (batch_size * feature_size)
        #     z1, z2 = self.projector(y1), self.projector(y2)
        #     return y1, y2, z1, z2
        # elif projector_type == 'simclr':
        #     y1, y2 = self.encoder(x1, projector_type='simclr'), self.encoder(x2, projector_type='simclr')  # (batch_size * feature_size)
        #     z1, z2 = self.projector_simclr(y1), self.projector_simclr(y2)
        #     return y1, y2, z1, z2

        y1_gap, y1_gmp, y1_att = self.encoder(x1, return_concat_combined=False)
        y2_gap, y2_gmp, y2_att = self.encoder(x2, return_concat_combined=False)

        if projector_type == 'vibcreg':
            projector = self.projector
        elif projector_type == 'simclr':
            projector = self.projector_simclr
        else:
            projector = None

        z1_gap = projector(y1_gap)
        z1_gmp = projector(y1_gmp)
        z1_att = projector(y1_att)
        z2_gap = projector(y2_gap)
        z2_gmp = projector(y2_gmp)
        z2_att = projector(y2_att)
        if return_y:
            return (y1_gap, y1_gmp, y1_att), (y2_gap, y2_gmp, y2_att), (z1_gap, z1_gmp, z1_att), (z2_gap, z2_gmp, z2_att)
        else:
            return (z1_gap, z1_gmp, z1_att), (z2_gap, z2_gmp, z2_att)


class contrastive_loss(nn.Module):
    def __init__(self, tau : float, normalize : bool):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, xi, xj, device):

        x = torch.cat((xi, xj), dim=0)

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
        norm_sum = norm_sum.to(device)
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss


class Utility_VIbCRegSimCLR(Utility_SSL):
    def __init__(self,
                 tau: float, normalize: bool,
                 lambda_vibcreg=25, mu_vibcreg=25, nu_vibcreg=200,
                 w_simclr=1,
                 length_sampling=False,
                 sample_len_ratios=None,
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
        self.w_simclr = w_simclr
        self.length_sampling = length_sampling
        self.sample_len_ratios = sample_len_ratios
        self.pool_type = kwargs.get('pool_type', None)

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
        # for subx_view1, subx_view2, label in data_loader:  # subx: (batch * n_channels * subseq_len)
        for x_view1, x_view2, label in data_loader:  # subx: (batch * n_channels * subseq_len)
            optimizer.zero_grad()

            L = 0
            # zs = []
            for i in range(2):
                # if self.length_sampling:
                seq_len = x_view1.shape[-1]
                # sample_len_ratio1 = np.random.uniform(*self.sample_len_ratios['view1'])
                sample_len_ratio1 = 0.5 if i == 0 else 1.0 #0.9 #0.9
                # sample_len_ratio2 = np.random.uniform(*self.sample_len_ratios['view2'])
                # subseq_lens = np.array([seq_len * sample_len_ratio1, seq_len * sample_len_ratio2]).astype(int)
                subseq_lens = np.array([seq_len * sample_len_ratio1, seq_len * sample_len_ratio1]).astype(int)
                rand_ts1 = 0 if seq_len == subseq_lens[0] else np.random.randint(0, seq_len - subseq_lens[0])
                rand_ts2 = 0 if seq_len == subseq_lens[1] else np.random.randint(0, seq_len - subseq_lens[1])
                subx_view1 = x_view1[:, :, rand_ts1:rand_ts1 + subseq_lens[0]]
                subx_view2 = x_view2[:, :, rand_ts2:rand_ts2 + subseq_lens[1]]

                # loss: vibcreg
                # y1, y2, z1, z2 = self.rl_model(subx_view1.to(self.device), subx_view2.to(self.device), 'vibcreg')
                # sim_loss = vibcreg_invariance_loss(z1, z2)
                # var_loss = vibcreg_var_loss(z1, z2)  # FcE loss
                # cov_loss = vibcreg_cov_loss(z1, z2)
                # L = self.lambda_vibcreg * sim_loss + self.mu_vibcreg * var_loss + self.nu_vibcreg * cov_loss

                (y1_gap, y1_gmp, y1_att), (y2_gap, y2_gmp, y2_att), \
                (z1_gap, z1_gmp, z1_att), (z2_gap, z2_gmp, z2_att) = self.rl_model(subx_view1.to(self.device), subx_view2.to(self.device), 'vibcreg', return_y=True)
                sim_loss = vibcreg_invariance_loss(z1_gap, z2_gap) + vibcreg_invariance_loss(z1_gmp, z2_gmp) + vibcreg_invariance_loss(z1_att, z2_att)
                sim_loss /= 3
                var_loss = vibcreg_var_loss(z1_gap, z2_gap) + vibcreg_var_loss(z1_gmp, z2_gmp) + vibcreg_var_loss(z1_att, z2_att)
                var_loss /= 3
                cov_loss = vibcreg_cov_loss(z1_gap, z2_gap) + vibcreg_cov_loss(z1_gmp, z2_gmp) + vibcreg_cov_loss(z1_att, z2_att)
                cov_loss /= 3
                L += self.lambda_vibcreg * sim_loss + self.mu_vibcreg * var_loss + self.nu_vibcreg * cov_loss

                # zs.append([z1_gap, z1_gmp, z1_att, z2_gap, z2_gmp, z2_att])
                # zs.append([z1_gap, z1_gmp, z2_gap, z2_gmp])

            # sim_loss_lg = 0
            # j = 0
            # for z_l, z_g in zip(zs[0], zs[1]):
            #     sim_loss_lg += vibcreg_invariance_loss(z_l, z_g, 'cos_sim')  # l: local, g: global
            #     j += 1
            # sim_loss_lg /= j
            # L += sim_loss_lg

            # loss: simclr
            # seq_len = x_view1.shape[-1]
            # sample_len_ratio1 = 0.9
            # subseq_lens = np.array([seq_len * sample_len_ratio1, seq_len * sample_len_ratio1]).astype(int)
            # rand_ts1 = 0 if seq_len == subseq_lens[0] else np.random.randint(0, seq_len - subseq_lens[0])
            # rand_ts2 = 0 if seq_len == subseq_lens[1] else np.random.randint(0, seq_len - subseq_lens[1])
            # subx_view1 = x_view1[:, :, rand_ts1:rand_ts1 + subseq_lens[0]]
            # subx_view2 = x_view2[:, :, rand_ts2:rand_ts2 + subseq_lens[1]]

            # (y1_gap, y1_gmp, y1_att), (y2_gap, y2_gmp, y2_att), \
            # (z1_gap, z1_gmp, z1_att), (z2_gap, z2_gmp, z2_att) = \
            #     self.rl_model(subx_view1.to(self.device), subx_view2.to(self.device), 'simclr', return_y=True)
            # simclr_loss = self.loss_func(z1_gap, z2_gap, self.device) + self.loss_func(z1_gmp, z2_gmp, self.device) #+ self.loss_func(z1_att, z2_att, self.device)
            # simclr_loss /= 2#3
            #
            # # y1, y2, z1, z2 = self.rl_model(subx_view1.to(self.device), subx_view2.to(self.device), 'simclr')
            # # simclr_loss = self.loss_func(z1, z2, self.device)
            # L += self.w_simclr * simclr_loss

            # if self.length_sampling:
            #     seq_len = x_view1.shape[-1]
            #     sample_len_ratio1 = np.random.uniform(0.9, 1.0)
            #     # sample_len_ratio2 = np.random.uniform(0.5, 1.0)
            #     sample_len_ratio2 = sample_len_ratio1
            #     subseq_lens = np.array([seq_len * sample_len_ratio1, seq_len * sample_len_ratio2]).astype(int)
            #     rand_ts1 = 0 if seq_len == subseq_lens[0] else np.random.randint(0, seq_len - subseq_lens[0])
            #     rand_ts2 = 0 if seq_len == subseq_lens[1] else np.random.randint(0, seq_len - subseq_lens[1])
            #     subx_view1 = x_view1[:, :, rand_ts1:rand_ts1 + subseq_lens[0]]
            #     subx_view2 = x_view2[:, :, rand_ts2:rand_ts2 + subseq_lens[1]]
            # y1, y2, z1, z2 = self.rl_model(subx_view1.to(self.device), subx_view2.to(self.device), 'simclr')
            # simclr_loss = self.loss_func(z1, z2, self.device)
            # L += self.w_simclr * simclr_loss

            # # loss: vibcreg-2
            # sim_loss2 = vibcreg_invariance_loss(z1, z2, 'mse')
            # var_loss2 = vibcreg_var_loss(z1, z2)  # FcE loss
            # cov_loss2 = vibcreg_cov_loss(z1, z2)
            # L += self.lambda_vibcreg * sim_loss2 + self.mu_vibcreg * var_loss2 + self.nu_vibcreg * cov_loss2
            # L += self.mu_vibcreg * var_loss2 + self.nu_vibcreg * cov_loss2

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
                         # 'sim_loss2': sim_loss2,
                         # 'var_loss2': var_loss2,
                         # 'cov_loss2': cov_loss2,
                         # 'simclr_loss': simclr_loss,
                         # 'sim_loss_lg': sim_loss_lg,
                         }
            # self.status_log_per_iter(status, y1, loss_hist)
            self.status_log_per_iter(status, torch.cat((y1_gap,y1_gmp,y1_att), dim=-1), loss_hist)

        if step == 0:
            return 0.
        else:
            return loss / step

    def _representation_for_validation(self, x):
        return super()._representation_for_validation(x)

