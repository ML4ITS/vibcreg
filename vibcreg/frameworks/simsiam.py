"""
# Define `Simple Siamese framework/network`

# References:
[1] X. Chen et al., 2020, "Exploring simple siamese representation learning".
[2] leaderj1001, "SimSiam", [github](https://github.com/leaderj1001/SimSiam)
"""
import torch.nn as nn
import wandb
from torch import relu

from vibcreg.backbone.resnet import ResNet1D, normalization_layer
from vibcreg.frameworks.framework_util_skeleton import Utility_SSL
from vibcreg.losses.invariance_loss import simsiam_cos_sim_loss


class Projector(nn.Module):
    def __init__(self, last_channels_enc, proj_hid, proj_out, norm_layer_type_proj):
        super().__init__()
        # define layers
        self.linear1 = nn.Linear(last_channels_enc, proj_hid)
        self.nl1 = normalization_layer(norm_layer_type_proj, proj_hid, dim=2)
        self.linear2 = nn.Linear(proj_hid, proj_hid)
        self.nl2 = normalization_layer(norm_layer_type_proj, proj_hid, dim=2)
        self.linear3 = nn.Linear(proj_hid, proj_out)
        self.nl3 = normalization_layer(norm_layer_type_proj, proj_out, dim=2)

    def forward(self, x):
        out = relu(self.nl1(self.linear1(x)))
        out = relu(self.nl2(self.linear2(out)))
        out = self.nl3(self.linear3(out))
        return out


class Predictor(nn.Module):
    def __init__(self, proj_out, pred_hid, pred_out, norm_layer_type_pred):
        super().__init__()
        self.linear1 = nn.Linear(proj_out, pred_hid)
        self.nl1 = normalization_layer(norm_layer_type_pred, pred_hid, dim=2)
        self.linear2 = nn.Linear(pred_hid, pred_out)

    def forward(self, x):
        out = relu(self.nl1(self.linear1(x)))
        out = self.linear2(out)
        return out


class SimSiam(nn.Module):
    def __init__(self, encoder: ResNet1D, last_channels_enc: int,
                 proj_hid_simsiam=2048, proj_out_simsiam=2048, norm_layer_type_proj_simsiam="BatchNorm",
                 pred_hid_simsiam=512, pred_out_simsiam=2048, norm_layer_type_pred_simsiam="BatchNorm", **kwargs):
        super().__init__()
        self.encoder = encoder
        self.projector = Projector(last_channels_enc, proj_hid_simsiam, proj_out_simsiam, norm_layer_type_proj_simsiam)
        self.predictor = Predictor(proj_out_simsiam, pred_hid_simsiam, pred_out_simsiam, norm_layer_type_pred_simsiam)

    def forward(self, x1, x2):
        y1, y2 = self.encoder(x1), self.encoder(x2)
        z1, z2 = self.projector(y1), self.projector(y2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return (z1, z2), (p1, p2)


class Utility_SimSiam(Utility_SSL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def wandb_watch(self):
        if self.use_wandb:
            wandb.watch(self.rl_model.module.encoder)
            wandb.watch(self.rl_model.module.projector)
            wandb.watch(self.rl_model.module.predictor)

    def representation_learning(self, data_loader, optimizer, status):
        """
        :param status: train / validate / test
        """
        self.rl_model.train() if status == "train" else self.rl_model.eval()

        loss, step = 0., 0
        for subx_view1, subx_view2, label in data_loader:  # subx: (batch * n_channels * subseq_len)
            optimizer.zero_grad()

            (z1, z2), (p1, p2) = self.rl_model(subx_view1.to(self.device), subx_view2.to(self.device))

            # loss
            L = simsiam_cos_sim_loss(p1, z2) / 2 + simsiam_cos_sim_loss(p2, z1) / 2  # symmetrical loss

            # weight update
            if status == "train":
                L.backward()
                optimizer.step()
                self.lr_scheduler.step()
                self.global_step += 1

            loss += L.item()
            step += 1

            # status log
            self.status_log_per_iter(status, z1)

        return loss / step

    def _representation_for_validation(self, x):
        return super()._representation_for_validation(x)
