"""
Autoregressive Predictive Coding (APC)

# Architecture:
1) Encoder: one GRU layer with hidden size of 512
2) (Forecasting) Predictor: (linear, relu, linear)

# Loss:
1) predictive coding (forecasting) loss

# Modifications
- better context (i.e., `kind == "mean+max"`)
- shallow prediction head (i.e., `Wk_fs` and `Wk_fs_2`)
"""
import wandb

import torch
import torch.nn as nn
from torch import relu

from vibcreg.backbone.apc_encoder import APCEncoder
from vibcreg.frameworks.framework_util_skeleton import Utility_SSL


class APC(nn.Module):
    def __init__(self, encoder: APCEncoder, forecast_pred_hid_apc=512, n_pred_steps_apc: list = (3, ), **kwargs):
        super().__init__()
        self.encoder = encoder
        self.forecast_pred_hid_apc = forecast_pred_hid_apc
        self.n_pred_steps_apc = n_pred_steps_apc
        self.in_channels_enc = self.encoder.in_channels_enc

        self.Wk_fs = nn.ModuleList([nn.Linear(self.encoder.h_size, self.forecast_pred_hid_apc) for _ in self.n_pred_steps_apc])
        self.Wk_fs_2 = nn.ModuleList([nn.Linear(self.forecast_pred_hid_apc, self.in_channels_enc) for _ in self.n_pred_steps_apc])

    def forward(self, x):
        """
        :param x: (B x C x (sub_)seq_len)
        """
        out = self.encoder(x)  # out (contexts == hidden states): (batch * seq_len * hidden_size)
        return out

    def forecast(self, c_t, idx):
        out = relu(self.Wk_fs[idx](c_t))  # `relu` possibly encourages the AR's hidden state to have positive features for meaningful input.
        out = self.Wk_fs_2[idx](out)
        return out


class Utility_APC(Utility_SSL):
    def __init__(self, n_pred_steps_apc, weight_on_pc_loss_apc=1., better_context_kind_apc="mean+max", **kwargs):
        super(Utility_APC, self).__init__(**kwargs)
        self.n_pred_steps_apc = n_pred_steps_apc
        self.weight_on_pc_loss_apc = weight_on_pc_loss_apc
        self.better_context_kind_apc = better_context_kind_apc
        self.in_channels_enc = self.rl_model.module.encoder.in_channels_enc

        self.criterion_mse = torch.nn.MSELoss()
        # self.criterion_l1_loss = torch.nn.L1Loss()

    def wandb_watch(self):
        if self.use_wandb:
            wandb.watch(self.rl_model.module.encoder)

    def representation_learning(self, data_loader, optimizer, status):
        self.rl_model.train() if status == "train" else self.rl_model.eval()

        loss, step = 0., 0
        for subx_view1, subx_view2, label in data_loader:  # subx: (batch * n_channels * subseq_len)
            optimizer.zero_grad()

            # loss: predictive coding loss (= forecasting loss)
            batch_size = subx_view1.shape[0]
            device_ = self.device
            subx_view1_T = torch.transpose(subx_view1, 1, 2).to(device_)  # (BxLx1); C:1 for univariate time series
            # - forecasting
            c1_ts = self.rl_model(subx_view1.to(device_))  # contexts; (B x L x h_size)
            pc_loss_f = 0.
            # for n_pred_step in range(1, self.n_pred_steps_apc+1):
            for i, n_pred_step in enumerate(self.n_pred_steps_apc):
                target_subx_view1 = subx_view1_T[:, n_pred_step:, :]  # (B x L-n_pred_step x 1)
                c1_ts_prime = c1_ts[:, :-n_pred_step, :]  # (B x L-n_pred_step x h_size)
                L_prime = c1_ts_prime.shape[1]  # L' = L-n_pred_steps
                pred_target_subx_view1 = torch.zeros(batch_size, L_prime, self.in_channels_enc).to(device_)
                for l in range(L_prime):
                    pred_target_subx_view1[:, l, :] = self.rl_model.module.forecast(c1_ts_prime[:, l, :], i)
                pc_loss_f_ = self.criterion_mse(pred_target_subx_view1, target_subx_view1)  # pc: predictive coding; f: forecast
                pc_loss_f += pc_loss_f_
            pc_loss = (pc_loss_f / len(self.n_pred_steps_apc))

            c1_t = c1_ts[:, -1, :]  # context; (B x h_size)

            L = self.weight_on_pc_loss_apc * pc_loss

            # weight Update
            if status == "train":
                L.backward()
                optimizer.step()
                self.lr_scheduler.step()
                self.global_step += 1

            loss += L.item()
            step += 1

            # status log
            self.status_log_per_iter(status, c1_t)

        return loss / step

    def _representation_for_validation(self, x):
        c_ts = self.rl_model.module.encoder(x.to(self.device)).detach().cpu()
        c_t = self.rl_model.module.encoder.compute_better_context(c_ts, kind=self.better_context_kind_apc)
        return c_t
