"""
CPC: Contrastive Predictive Coding
The paper [4] presents that CPC results in the best performance on the ECG dataset over BYOL and SimCLR.

# Remarks
- It doesn't use a ResNet backbone. It has its own backbone architecture (i.e., downsampling encoder).

# References:
[1] A. Van Den Oord et al., 2018, "Representation learning with contrastive predictive coding"
[2] jefflai108, "Contrastive-Predictive-Coding-PyTorch", [Github](https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch)
[3] K. He et al., 2019, "Momentum contrast for unsupervised visual representation learning"; Regarding 'InfoNCE', 'temperature'
[4] T. Mehari and N. Strodthoff, 2021, "Self-supervised representation learning from 12-lead ECG data"; Regarding 'deeper predictor'
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from vibcreg.backbone.downsampling_cnn import DownsamplingCNN
from vibcreg.frameworks.framework_util_skeleton import Utility_SSL


class AR(nn.Module):
    """
    Autoregressive model to yield 'c_t' (context).
    In the original CPC, GRU is used.

    References:
    [1] PyTorch, "GRU", https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
    """
    def __init__(self, enc_hid_channels_cpc):
        super().__init__()
        self.n_layers = 1

        # define layers
        in_size = enc_hid_channels_cpc  # number of expected features
        self.h_size = in_size // 2
        self.ar = nn.GRU(in_size, self.h_size, self.n_layers, batch_first=True)

        # define states
        self.h_n = None  # hidden state

    def initialize_hidden_state(self, batch_size, device):
        """Initializes the hidden state of the autoregressive model."""
        self.h_n = torch.zeros(self.n_layers, batch_size, self.h_size, requires_grad=True).to(device)

    def forward(self, x):
        self.ar.flatten_parameters()
        out, self.h_n = self.ar(x, self.h_n)  # out: (batch * seq_len * hidden_size)
        return out


class Predictor(nn.Module):
    """It receives 'c_t' and make a prediction to predict 'z_{t+k}' """
    def __init__(self, enc_hid_channels_cpc, max_pred_step):
        super().__init__()
        self.in_size = enc_hid_channels_cpc // 2
        self.out_size = enc_hid_channels_cpc
        self.max_pred_step = max_pred_step

        # define Wk
        self.Wk = nn.ModuleList([nn.Linear(self.in_size, self.out_size) for _ in range(self.max_pred_step)])


class CPC(nn.Module):
    def __init__(self, encoder: DownsamplingCNN, max_pred_step=4, **kwargs):
        super().__init__()
        self.max_pred_step = max_pred_step
        self.enc_hid_channels_cpc = encoder.enc_hid_channels_cpc

        # define modules
        self.encoder = encoder
        self.ar = AR(self.enc_hid_channels_cpc)  # autoregressive model
        self.predictor = Predictor(self.enc_hid_channels_cpc, self.max_pred_step)

    def forward(self, x):
        """
        s.t. PTB-XL (12 leads)
        x: time series (batch * 12 * seq_len); {12: n_channels}; e.g. (8, 12, 20480)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[-1]
        downsampling_factor = self.encoder.downsampling_factor
        device_ = x.get_device()

        # Get a random timestep (belongs to the representation space)
        rand_t = torch.randint(int(seq_len // downsampling_factor) - self.max_pred_step, size=(1,)).item()

        # Downsample `x` to `z` with the encoder
        z = self.encoder(x)  # (batch * n_channels * reduced_seq_len); e.g. (8, 512, 128)
        z = z.transpose(1, 2)  # (batch * reduced_seq_len * n_channels); e.g. (8, 128, 512)

        # Compute [z_{rand_t+1}, z_{rand_t+2}, ..., z_{rand_t + max_pred_step}]
        z_future = torch.zeros(batch_size, self.max_pred_step, self.enc_hid_channels_cpc).to(device_)  # e.g. (8, 12, 512)
        for k in np.arange(1, self.max_pred_step+1):
            i = k-1
            z_future[:, i, :] = z[:, rand_t+k, :]  # e.g. (8, 512)

        # Compute `c_t`
        forward_seq = z[:, :rand_t+1, :]  # e.g. (8, 100, 512)
        self.ar.initialize_hidden_state(batch_size, device_)
        out = self.ar(forward_seq)  # e.g. (8, 100, 256)
        c_t = out[:, rand_t, :]  # e.g. (8, 256); 'context'

        # Compute [zhat_{rand_t+1}, zhat_{rand_t+2}, ..., zhat_{rand_t + max_pred_step}]
        zhat_future = torch.zeros(batch_size, self.max_pred_step, self.enc_hid_channels_cpc).to(device_)  # e.g. (8, 12, 512)
        for i in np.arange(self.max_pred_step):
            zhat_future[:, i, :] = self.predictor.Wk[i](c_t)  # c_t*Wk[i]^T (8, 512); Wk[i] (512, 256)
        return z_future, zhat_future, c_t

    def compute_infoNCE(self, z_future, zhat_future):
        batch_size = z_future.shape[0]
        device_ = z_future.device

        nce = 0.
        acc = {f'pred_step-{k}': 0. for k in np.arange(1, self.max_pred_step+1)}
        for i in np.arange(self.max_pred_step):
            cross_corr = torch.mm(z_future[:, i, :], zhat_future[:, i, :].T)  # e.g. (8, 512) x (512, 8) -> cross-correlation matrix; The diagonal elements are the true classes.
            # cross_corr /= self.temperature_cpc
            nce += -1 * torch.mean(torch.diag(F.log_softmax(cross_corr, dim=1)))
            correct = torch.mean(torch.eq(torch.argmax(F.softmax(cross_corr, dim=1), dim=1), torch.arange(batch_size).to(device_)).float())
            acc[f'pred_step-{i+1}'] = correct.item()
        nce /= self.max_pred_step
        return nce, acc

    def get_z_and_c_on_x(self, x, device):
        """
        Propagate `x` via Encoder and AR, and obtain 'z' and 'c' (context)'.
        - 'x': (batch * 1 * entire_seq_len)
        - 'z' is the output from Encoder; (batch * n_channels * reduced_seq_len)
        - 'c' is the output from AR; (batch * feature_size_ar)
        """
        z = self.encoder(x)  # (batch * n_channels * reduced_seq_len)
        zT = z.transpose(1, 2)  # (batch * reduced_seq_len * n_channels)
        batch_size = x.shape[0]
        self.ar.initialize_hidden_state(batch_size, device)
        out_ar = self.ar(zT)  # (batch * reduced_seq_len * feature_size_ar)
        c_t = out_ar[:, -1, :]  # (final) context; (batch * feature_size_ar)
        return z, c_t


class Utility_CPC(Utility_SSL):
    def __init__(self, **kwargs):
        super(Utility_CPC, self).__init__(**kwargs)
        self.acc_cpc = None  # accuracy from `compute_infoNCE(..)`.

    def wandb_watch(self):
        if self.use_wandb:
            wandb.watch(self.rl_model.module.encoder)
            wandb.watch(self.rl_model.module.ar)
            wandb.watch(self.rl_model.module.predictor)

    def representation_learning(self, data_loader, optimizer, status):
        """
        :param status: train / validate / test
        """
        self.rl_model.train() if status == "train" else self.rl_model.eval()

        loss, step = 0., 0
        for subx_view1, subx_view2, label in data_loader:  # subx: (batch * n_channels * subseq_len)
            optimizer.zero_grad()

            # loss
            z_future, zhat_future, c_t = self.rl_model(subx_view1.to(self.device))
            nce_loss, self.acc_cpc = self.rl_model.module.compute_infoNCE(z_future, zhat_future)

            # weight update
            if status == "train":
                nce_loss.backward()
                optimizer.step()
                self.lr_scheduler.step()
                self.global_step += 1

            loss += nce_loss.item()
            step += 1

            # status log
            self.status_log_per_iter(status, c_t)

        return loss / step

    def _representation_for_validation(self, x):
        _, _, c_t = self.rl_model(x.to(self.device))
        c_t = c_t.detach().cpu()
        return c_t

    def log_accuracy_cpc(self):
        """
        There are 'n (= max_pred_step)' prediction steps (e.g., 12), thus there exists accuracy per step.
        """
        if not self.use_wandb:
            return None

        plt.figure(figsize=(6, 4))

        plt.plot(np.arange(1, len(self.acc_cpc) + 1), self.acc_cpc.values(), 'o-')
        plt.xlim(0, ); plt.ylim(-0.03, 1.03)

        plt.xlabel('Prediction step', size=14); plt.ylabel('Accuracy', size=14)
        plt.grid(); plt.tight_layout()
        wandb.log({f'Accuracy-CPC-ep_{self.epoch}': [wandb.Image(plt, caption=f'')]})
        plt.close()
