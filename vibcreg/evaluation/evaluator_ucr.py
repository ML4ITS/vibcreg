import numpy as np
import matplotlib.pyplot as plt
import wandb

import torch
import torch.nn as nn
from torch import relu
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from vibcreg.normalization.iter_norm import IterNorm
from vibcreg.lr_scheduler.cosine_annealing_lr import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from vibcreg.frameworks.vibcreg_ import VIbCReg
from vibcreg.frameworks.barlow_twins import BarlowTwins
from vibcreg.frameworks.simsiam import SimSiam
from vibcreg.frameworks.rand_init import RandInit
from vibcreg.frameworks.cpc import CPC
from vibcreg.frameworks.apc import APC

from vibcreg.evaluation.evaluator_skeleton import Evaluator


class EvaluatorUCR(Evaluator):
    def __init__(self, **kwargs):
        super(EvaluatorUCR, self).__init__(**kwargs)

    def _set_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def _clf_in_size(self, framework_type, **kwargs) -> int:
        if framework_type == 'cpc':
            in_size = self.rl_model.module.ar.ar.input_size
        elif framework_type == "apc":
            better_context_kind_apc = kwargs.get("better_context_kind_apc", None)
            mul = len(better_context_kind_apc.split("+"))
            in_size = self.encoder.module.rnn.hidden_size * mul
        else:  # if `encoder` is `ResNet1D`
            in_size = self.encoder.module.res_blocks[-1].conv_1x1.out_channels
        return in_size

    def _clf_out_size(self):
        return len(np.unique(self.train_data_loader.dataset.Y))

    def _adjust_label(self, label):
        """
        adjust `label` such that it'd have ordered class indices such as [0, 1, 2] instead of [-1, 1] or [1, 2, 3]
        """
        label = label.view(-1).long()
        if torch.min(label) == -1:  # `Wafer` has labels of -1 and 1.
            label = relu(label)  # make -1 zero, and keep 1 as it is.
        else:
            label = label - torch.min(label)
        return label

    def _out_from_encoder_classifier(self, x, **kwargs):
        """
        returns output from an encoder-classifier.
        """
        y = self.encoder(x)
        if kwargs.get("framework_type") == "apc":
            better_context_kind_apc = kwargs.get("better_context_kind_apc", None)
            y = self.encoder.module.compute_better_context(y, kind=better_context_kind_apc)

        out = self.classifier(y)  # (batch * 1)
        return out

    def _train_eval_setting_during_fit(self, status):
        """
        set the .train() / .eval() settings during fit(..) for `self.encoder` and `self.classifier`.
        """
        if status == "train":
            # encoder
            if (self.framework_type == 'supervised') or (self.framework_type == "apc"):
                self.encoder.train()
            elif self.evaluation_type == "fine_tuning_evaluation":
                self._enable_bn_stat_update_only(self.encoder)
            else:
                self.encoder.eval()
            # classifier
            self.classifier.train()

        elif (status == "validate") or (status == "test"):
            self.encoder.eval()
            if self.framework_type == "apc":
                self.encoder.train()
            self.classifier.eval()

