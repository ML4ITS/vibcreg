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
        self.nl3 = normalization_layer('IterNorm', proj_out,
                                       dim=2) if self.add_IterN_at_the_last_in_proj_vibcreg else None

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
        self.projector = Projector(last_channels_enc, 4 * last_channels_enc,
                                   4 * last_channels_enc)  # Projector(last_channels_enc, proj_hid_vibcreg, proj_out_vibcreg)
        # self.projector_simclr = ProjectorSimCLR(last_channels_enc, proj_hid_vibcreg, proj_out_vibcreg)

    def forward(self, x1, x2, projector_type: str = 'vibcreg', return_y=False, one_sided_partial_mask=0.):
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

        (gap_y1, gmp_y1, amp_y1), (gap_cy1, gmp_cy1, amp_cy1) = self.encoder(x1, return_concat_combined=False)
        (gap_y2, gmp_y2, amp_y2), (gap_cy2, gmp_cy2, amp_cy2) = self.encoder(x2, return_concat_combined=False)

        if projector_type == 'vibcreg':
            projector = self.projector
        elif projector_type == 'simclr':
            projector = self.projector_simclr
        else:
            projector = None

        z1_gap = projector(gap_y1)
        z1_gmp = projector(gmp_y1)
        z1_amp = projector(amp_y1)

        z2_gap = projector(gap_y2)
        z2_gmp = projector(gmp_y2)
        z2_amp = projector(amp_y2)

        cz1_gap = projector(gap_cy1)
        cz1_gmp = projector(gmp_cy1)
        cz1_amp = projector(amp_cy1)

        cz2_gap = projector(gap_cy2)
        cz2_gmp = projector(gmp_cy2)
        cz2_amp = projector(amp_cy2)

        if return_y:
            return (gap_y1, gmp_y1, amp_y1), (gap_y2, gmp_y2, amp_y2), \
                   (gap_cy1, gmp_cy1, amp_cy1), (gap_cy2, gmp_cy2, amp_cy2), \
                   (z1_gap, z1_gmp, z1_amp), (z2_gap, z2_gmp, z2_amp), \
                   (cz1_gap, cz1_gmp, cz1_amp), (cz2_gap, cz2_gmp, cz2_amp)
        else:
            return (z1_gap, z1_gmp, z1_amp), (z2_gap, z2_gmp, z2_amp), \
                   (cz1_gap, cz1_gmp, cz1_amp), (cz2_gap, cz2_gmp, cz2_amp)


class contrastive_loss(nn.Module):
    def __init__(self, tau: float, normalize: bool):
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
                 one_sided_partial_mask=0.,
                 within_augs=[],
                 len_ratios=(0.5, 1.),
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
        self.one_sided_partial_mask = one_sided_partial_mask
        self.within_augs = within_augs
        self.len_ratios = len_ratios

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

    def amplitude_resize(self, subx_views, AmpR_rate):
        """
        :param subx_view: (B, C, L)
        """
        new_subx_views = []
        B, n_channels = subx_views[0].shape[0], subx_views[0].shape[1]
        for i in range(len(subx_views)):
            # mul_AmpR = 1 + np.random.uniform(-AmpR_rate, AmpR_rate, size=(B, n_channels, 1))
            mul_AmpR = 1 + np.random.normal(0., AmpR_rate, size=(B, n_channels, 1))
            new_subx_view = subx_views[i] * torch.from_numpy(mul_AmpR).float()
            new_subx_views.append(new_subx_view)

        if len(new_subx_views) == 1:
            new_subx_views = new_subx_views[0]
        return new_subx_views

    def vertical_shift(self, subx_views, Vshift_rate):
        """
        :param subx_view: (B, C, L)
        """
        new_subx_views = []
        B, n_channels = subx_views[0].shape[0], subx_views[0].shape[1]
        for i in range(len(subx_views)):
            std_x = torch.std(subx_views[i], dim=-1, keepdim=True)  # (B, C, 1)
            vshift_mag = std_x * torch.from_numpy(np.random.uniform(-Vshift_rate, Vshift_rate, size=(B, n_channels, 1)))
            vshift_mag = vshift_mag.float()
            new_subx_view = subx_views[i] + vshift_mag
            new_subx_views.append(new_subx_view)

        if len(new_subx_views) == 1:
            new_subx_views = new_subx_views[0]
        return new_subx_views

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
            sim_loss = 0
            var_loss = 0
            cov_loss = 0
            len_ratios = [(r, r) for r in self.len_ratios]
            # for len_ratio in len_ratios:
            for len_ratio in len_ratios:
                seq_len = x_view1.shape[-1]
                sample_len_ratio1, sample_len_ratio2 = len_ratio[0], len_ratio[1]
                # subseq_lens = np.array([seq_len * sample_len_ratio1, seq_len * sample_len_ratio1]).astype(int)
                subseq_lens = np.array([seq_len * sample_len_ratio1, seq_len * sample_len_ratio2]).astype(int)
                # if len_ratio != 1.0:
                subx_view1 = torch.zeros(x_view1.shape[0], x_view1.shape[1], subseq_lens[0]).float()
                subx_view2 = torch.zeros(x_view2.shape[0], x_view2.shape[1],
                                         subseq_lens[1]).float()  # torch.clone(subx_view1)
                for b in range(x_view1.shape[0]):
                    rand_ts1 = 0 if seq_len == subseq_lens[0] else np.random.randint(0, seq_len - subseq_lens[0])
                    rand_ts2 = 0 if seq_len == subseq_lens[1] else np.random.randint(0, seq_len - subseq_lens[1])
                    subx_view1[b] = x_view1[b, :, rand_ts1:rand_ts1 + subseq_lens[0]]
                    subx_view2[b] = x_view2[b, :, rand_ts2:rand_ts2 + subseq_lens[1]]
                # else:
                #     subx_view1 = x_view1
                #     subx_view2 = x_view2

                # loss: vibcreg
                # if len_ratio == 1.0:
                #     (y1_gap, y1_gmp, y1_att), (y2_gap, y2_gmp, y2_att), \
                #     (z1_gap, z1_gmp, z1_att), (z2_gap, z2_gmp, z2_att) = self.rl_model(subx_view1.to(self.device),
                #                                                                        subx_view2.to(self.device),
                #                                                                        'vibcreg',
                #                                                                        return_y=True,
                #                                                                        one_sided_partial_mask=self.one_sided_partial_mask)
                #     sim_loss_mm = vibcreg_invariance_loss(z1_gap.detach(), z2_gap) \
                #                   + vibcreg_invariance_loss(z1_gmp.detach(), z2_gmp) \
                #                   + vibcreg_invariance_loss(z1_att.detach(), z2_att)
                #     sim_loss_mm /= 3
                #     var_loss_mm = vibcreg_var_loss(z1_gap, z1_gap) / 2 \
                #                   + vibcreg_var_loss(z1_gmp, z1_gmp) / 2 \
                #                   + vibcreg_var_loss(z1_att, z1_att) / 2
                #     var_loss_mm /= 3
                #     cov_loss_mm = vibcreg_cov_loss(z1_gap, z1_gap) / 2 \
                #                   + vibcreg_cov_loss(z1_gmp, z1_gmp) / 2 \
                #                   + vibcreg_cov_loss(z1_att, z1_att) / 2
                #     cov_loss_mm /= 3
                #     L += self.lambda_vibcreg * sim_loss_mm + self.mu_vibcreg * var_loss_mm + self.nu_vibcreg * cov_loss_mm
                # else:

                # augmentations
                # subx_view1, subx_view2 = self.vertical_shift([subx_view1, subx_view2], Vshift_rate=0.5)
                for aug in self.within_augs:
                    if aug == 'AmpR':
                        subx_view1, subx_view2 = self.amplitude_resize([subx_view1, subx_view2], AmpR_rate=0.1)

                (gap_y1, gmp_y1, amp_y1), (gap_y2, gmp_y2, amp_y2), \
                (gap_cy1, gmp_cy1, amp_cy1), (gap_cy2, gmp_cy2, amp_cy2), \
                (z1_gap, z1_gmp, z1_amp), (z2_gap, z2_gmp, z2_amp), \
                (cz1_gap, cz1_gmp, cz1_amp), (cz2_gap, cz2_gmp, cz2_amp) = self.rl_model(subx_view1.to(self.device),
                                                                                         subx_view2.to(self.device),
                                                                                         'vibcreg',
                                                                                         return_y=True,
                                                                                         one_sided_partial_mask=self.one_sided_partial_mask)

                sim_loss += vibcreg_invariance_loss(z1_gap, z2_gap) \
                            + vibcreg_invariance_loss(z1_gmp, z2_gmp) \
                            + vibcreg_invariance_loss(z1_amp, z2_amp)
                var_loss += vibcreg_var_loss(z1_gap, z2_gap) \
                            + vibcreg_var_loss(z1_gmp, z2_gmp) \
                            + vibcreg_var_loss(z1_amp, z2_amp)
                cov_loss += vibcreg_cov_loss(z1_gap, z2_gap) \
                            + vibcreg_cov_loss(z1_gmp, z2_gmp) \
                            + vibcreg_cov_loss(z1_amp, z2_amp)

                sim_loss += vibcreg_invariance_loss(cz1_gap, cz2_gap) \
                            + vibcreg_invariance_loss(cz1_gmp, cz2_gmp) \
                            + vibcreg_invariance_loss(cz1_amp, cz2_amp)
                var_loss += vibcreg_var_loss(cz1_gap, cz2_gap) \
                            + vibcreg_var_loss(cz1_gmp, cz2_gmp) \
                            + vibcreg_var_loss(cz1_amp, cz2_amp)
                cov_loss += vibcreg_cov_loss(cz1_gap, cz2_gap) \
                            + vibcreg_cov_loss(cz1_gmp, cz2_gmp) \
                            + vibcreg_cov_loss(cz1_amp, cz2_amp)

            L += self.lambda_vibcreg * sim_loss + self.mu_vibcreg * var_loss + self.nu_vibcreg * cov_loss

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
                         # 'sim_loss_mm': sim_loss_mm,
                         # 'var_loss_mm': var_loss_mm,
                         # 'cov_loss_mm': cov_loss_mm,
                         # 'sim_loss2': sim_loss2,
                         # 'var_loss2': var_loss2,
                         # 'cov_loss2': cov_loss2,
                         # 'simclr_loss': simclr_loss,
                         # 'sim_loss_lg': sim_loss_lg,
                         }
            # self.status_log_per_iter(status, y1, loss_hist)
            self.status_log_per_iter(status, torch.cat((gap_y1, gmp_y1, amp_y1), dim=-1), loss_hist)

        if step == 0:
            return 0.
        else:
            return loss / step

    def _representation_for_validation(self, x):
        return super()._representation_for_validation(x)
