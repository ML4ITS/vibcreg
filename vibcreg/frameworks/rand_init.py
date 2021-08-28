"""
`Random Init` denotes a "frozen randomly-initialized encoder".
"""
import torch.nn as nn

from vibcreg.backbone.resnet import ResNet1D
from vibcreg.frameworks.framework_util_skeleton import Utility_SSL


class RandInit(nn.Module):
    def __init__(self, encoder: ResNet1D):
        super().__init__()
        self.encoder = encoder


class Utility_RandInit(Utility_SSL):
    def __init__(self, **kwargs):
        super(Utility_RandInit, self).__init__(**kwargs)

    def wandb_watch(self):
        pass

    def representation_learning(self, data_loader, optimizer, status):
        """
        :param status: train / validate / test
        """
        self.rl_model.train() if status == "train" else self.rl_model.eval()

        # for subx_view1, subx_view2, label in data_loader:  # subx: (batch * n_channels * subseq_len)
        #     break

        return 0.

    def _representation_for_validation(self, x):
        return super()._representation_for_validation(x)
