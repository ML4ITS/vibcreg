import torch
import torch.nn as nn


class APCEncoder(nn.Module):
    """
    Encoder of APC.
    It's just a 1-GRU model.
    """
    def __init__(self, in_channels_enc, ar_hid_apc=512, n_ar_layers_apc=1, **kwargs):
        super().__init__()
        self.in_channels_enc = in_channels_enc
        self.ar_hid_apc = ar_hid_apc
        self.n_ar_layers_apc = n_ar_layers_apc

        # define layers
        self.in_size = self.in_channels_enc  # 1 for univariate time series
        self.h_size = self.ar_hid_apc
        self.rnn = nn.GRU(self.in_size, self.h_size, self.n_ar_layers_apc, batch_first=True)

        # define states
        self.h_n = None  # hidden state

    def _init_hid_state(self, batch_size, device):
        self.h_n = torch.zeros(self.n_ar_layers_apc, batch_size, self.h_size, requires_grad=True).to(device)

    def forward(self, x):
        """
        :param x: (B x C x (sub_)seq_len)
        """
        batch_size = x.shape[0]
        self._init_hid_state(batch_size, x.get_device())

        xT = torch.transpose(x, 1, 2)  # (B x (sub_)seq_len x C)
        out, self.h_n = self.rnn(xT, self.h_n)  # out: (batch * seq_len * hidden_size)
        return out

    @staticmethod
    def compute_better_context(c_ts, kind):
        """
        :param c_ts (B x L x h_size)
        :param kind: last | mean | max | last+mean+max | mean+max
        follows the concat pooling layer from [T. Mehari, 2021, "Self-supervised representation learning from 12-lead ECG data"]

        It forms by concatenating
            1. the maximum of all RNN outputs,
            2. the mean of all RNN outputs,
            3. the RNN output corresponding to the final step.
        """
        mean_context = c_ts.mean(dim=1)  # (B x h_size)

        max_context = c_ts.max(dim=1).values  # (B x h_size)
        # B, H = c_ts.shape[0], c_ts.shape[2]
        # ind0 = (np.ones((B, H)) * np.arange(B).reshape(-1, 1)).astype(np.int)
        # ind1 = torch.abs(c_ts).max(dim=1).indices
        # max_context = c_ts[ind0, ind1, np.arange(H)]

        last_context = c_ts[:, -1, :]  # (B x h_size)

        if kind == "last":
            return last_context
        elif kind == "mean":
            return mean_context
        elif kind == "max":
            return max_context
        elif kind == "last+mean+max":
            return torch.cat((last_context, mean_context, max_context), dim=-1)
        elif kind == "mean+max":
            return torch.cat((mean_context, max_context), dim=-1)
        else:
            raise ValueError("invalid `kind`")