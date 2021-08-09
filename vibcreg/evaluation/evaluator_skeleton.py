from abc import ABC, abstractmethod
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


def update_config(cf, **kwargs):
    for k, v in kwargs.items():
        cf[k] = v


class WBLogger(object):
    # [1] https://docs.wandb.ai/library/log
    def __init__(self):
        self.best_train_loss = None
        self.best_val_loss = None
        self.best_test_loss = None
        self.best_test_acc = None
        self.best_test_macroAUC = None
        self.count = 0

    @staticmethod
    def log(**kwargs):
        log_content = {k: v for k, v in kwargs.items()}
        wandb.log(log_content)

    @staticmethod
    def main_log(epoch, train_loss, val_loss, test_loss, **kwargs):
        test_acc, test_macroAUC = kwargs.get("test_acc", None), kwargs.get("test_macroAUC", None)
        keys = ['epoch', 'train_loss', 'val_loss', 'test_loss']
        vals = [epoch, train_loss, val_loss, test_loss]
        if test_acc:
            keys = keys + ["test_acc"]
            vals = vals + [test_acc]
        if test_macroAUC:
            keys = keys + ["test_macroAUC"]
            vals = vals + [test_macroAUC]
        log_content = {k: v for k, v in zip(keys, vals)}
        wandb.log(log_content)

    def main_summary(self, train_loss, val_loss, test_loss, **kwargs):
        """
        Note that `wand.run.summary` must be run at every step along with `wand.log(..)`.
        """
        if self.best_train_loss is None:
            self.best_train_loss = train_loss
            self.best_val_loss = val_loss
            self.best_test_loss = test_loss
            self.best_test_acc = kwargs.get("test_acc", None)
            self.best_test_macroAUC = kwargs.get("test_macroAUC", None)

        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss
            wandb.run.summary['train_loss'] = self.best_train_loss

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            wandb.run.summary['val_loss'] = self.best_val_loss

        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            wandb.run.summary['test_loss'] = self.best_test_loss

        test_acc = kwargs.get("test_acc")
        if test_acc and (test_acc > self.best_test_acc):
            self.best_test_acc = test_acc
            wandb.run.summary['test_acc'] = self.best_test_acc

        test_macroAUC = kwargs.get("test_macroAUC")
        if test_macroAUC and (test_macroAUC > self.best_test_macroAUC):
            self.best_test_macroAUC = test_macroAUC
            wandb.run.summary['test_macroAUC'] = self.best_test_acc

        self.count += 1


def get_children(model: nn.Module):
    """
    :param model: pytorch model
    :return: children of `model`
    """
    # [1] https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
        # look for children from children... to the last child!
        for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


class Evaluator(ABC):
    """
    It is used for both 'linear evaluation' and 'fine-tuning evaluation'.
    """
    def __init__(self, cf, train_data_loader, val_data_loader, test_data_loader, **kwargs):
        self.cf = cf
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

        # params
        self.project_name = cf.get("project_name", None)
        self.evaluation_type = cf.get("evaluation_type", None)
        self.device_ids = cf.get("device_ids", None)
        self.dataset_name = cf.get("dataset_name", None)
        self.framework_type = cf.get("framework_type", None)
        self.use_wandb = cf.get("use_wandb", None)
        self.loading_checkpoint_fname = cf.get("loading_checkpoint_fname", None)

        self.fine_tune = self._set_fine_tune()
        self.criterion = self._set_criterion()

        self.encoder = None
        self.rl_model = None
        self.ar_cpc = None
        self.classifier = None
        self.wb = None
        self.lr_scheduler = None
        self.epoch = None

    def _set_fine_tune(self) -> bool:
        if self.evaluation_type == "linear_evaluation":
            fine_tune = False
        elif self.evaluation_type == "fine_tuning_evaluation":
            fine_tune = True
        else:
            raise ValueError("invalid `evaluation_type`")
        return fine_tune

    @abstractmethod
    def _set_criterion(self):
        """
        set `criterion` that's suitable for your dataset.
        e.g.,
        - nn.CrossEntropyLoss() for the UCR datasets.
        - nn.BCEWithLogitsLoss() for the PTB-XL.
        """
        if self.dataset_name == "UCR":
            criterion = nn.CrossEntropyLoss()
        elif self.dataset_name == "PTB-XL":
            criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Define `self.criterion` for your `dataset_name`.")
        return criterion

    @staticmethod
    def _freeze(model):
        """
        freeze learnable parameters in a model.
        """
        for param in model.parameters():
            param.requires_grad = False

    def load_model(self, encoder):
        # load
        if self.framework_type == "supervised":
            self.encoder = encoder
            self.encoder = nn.DataParallel(self.encoder, device_ids=self.device_ids)
        else:
            if self.framework_type == "rand_init":
                self.rl_model = RandInit(encoder)
            elif (self.framework_type == "vibcreg") or (self.framework_type == "vbibcreg"):
                self.rl_model = VIbCReg(encoder, encoder.last_channels_enc, **self.cf)
            elif self.framework_type == "barlow_twins":
                self.rl_model = BarlowTwins(encoder, encoder.last_channels_enc, **self.cf)
            elif self.framework_type == "simsiam":
                self.rl_model = SimSiam(encoder, encoder.last_channels_enc, **self.cf)
            elif self.framework_type == "cpc":
                self.rl_model = CPC(encoder, **self.cf)
                self.ar_cpc = nn.DataParallel(self.rl_model.ar, device_ids=self.device_ids)
            elif self.framework_type == "apc":
                self.rl_model = APC(encoder, **self.cf)
            else:
                raise ValueError("invalid `framework_type`")

            checkpoint = torch.load(self.loading_checkpoint_fname)
            self.rl_model.load_state_dict(checkpoint['model_state_dict'])
            self.encoder = nn.DataParallel(self.rl_model.encoder, device_ids=self.device_ids)
            self._freeze(self.encoder) if not self.fine_tune else None

    @abstractmethod
    def _clf_in_size(self, **kwargs) -> int:
        """
        get an input size of a classification head.
        """
        if self.framework_type == 'cpc':
            in_size = self.rl_model.module.ar.ar.input_size
        elif self.framework_type == "apc":
            better_context_kind_apc = kwargs.get("better_context_kind_apc", None)
            mul = len(better_context_kind_apc.split("+"))
            in_size = self.encoder.module.rnn.hidden_size * mul
        else:
            # if `encoder` is `ResNet1D`.
            in_size = self.encoder.module.res_blocks[-1].conv_1x1.out_channels
        return in_size

    @abstractmethod
    def _clf_out_size(self) -> int:
        """
        get a number of unique labels to determine an output size of a classification head (= linear classifier).
        e.g.,
        :return 71 for the PTB-XL.
        """
        pass

    def build_classifier(self, **kwargs):
        in_size = self._clf_in_size(**kwargs)
        out_size = self._clf_out_size()

        # build a classifier
        self.classifier = nn.Sequential(nn.Linear(in_size, out_size))
        self.classifier = nn.DataParallel(self.classifier, device_ids=self.device_ids)

    def init_wandb(self, **kwargs):
        # set `run_name`
        run_name = f"{self.framework_type}"
        if self.dataset_name == "UCR":
            ucr_dataset_name = kwargs.get("ucr_dataset_name", None)
            run_name = f"{ucr_dataset_name}-" + run_name

        if self.loading_checkpoint_fname:
            ep = int(self.loading_checkpoint_fname.split("ep_")[1].split(".pth")[0])
            run_name = run_name + f"-ep_{ep}"

        if self.fine_tune and kwargs.get("train_data_ratio", None):
            train_data_ratio = kwargs.get("train_data_ratio", None) / 8 * 1000  # 8: 8 training folds; 1000 to make it percentage.
            run_name = run_name + f"-{train_data_ratio}"

        # set `project_name`
        if not self.fine_tune:
            project_name = self.project_name + "-LE"  # linear evaluation
        else:
            project_name = self.project_name + "-FtE"  # fine-tuning evaluation

        # initialize wandb
        if self.use_wandb:
            self.wb = wandb.init(project=project_name, config=self.cf, name=run_name)
        else:
            self.wb = None

    def wandb_watch(self):
        if not self.use_wandb:
            return None
        wandb.watch(self.encoder)
        wandb.watch(self.classifier)

    def finish_wandb(self):
        self.wb.finish() if self.use_wandb else None

    def setup_lr_scheduler(self, optimizer, kind: str, batch_size: int, n_epochs_ev: dict, **kwargs):
        if kind == "CosineAnnealingLR":
            train_dataset_size = kwargs.get("train_dataset_size", None)
            n_gpus = len(self.device_ids)
            self.lr_scheduler = CosineAnnealingLR(optimizer, train_dataset_size, n_gpus, batch_size, n_epochs_ev[self.evaluation_type]).get_lr_scheduler()
        else:
            raise ValueError("unavailable name for `lr_scheduler`.")

    def trainable_params(self, lr_clf_ev: float, lr_enc_ev: float, **kwargs):
        """
        returns 'trainable parameters'
        """
        if (self.framework_type == "supervised") or (self.fine_tune is True):
            params = [{"params": self.classifier.parameters(), "lr": lr_clf_ev},
                      {"params": self.encoder.parameters(), "lr": lr_enc_ev}]
        else:  # linear evaluation
            params = [{"params": self.classifier.parameters(), "lr": lr_enc_ev}]
        return params

    @abstractmethod
    def _adjust_label(self, label):
        """
        receives `label` and processes it if necessary, and returns it.
        """
        pass

    def _out_from_encoder_classifier(self, x, **kwargs):
        """
        returns output from an encoder-classifier.
        """
        y = self.encoder(x)
        if self.framework_type == "apc":
            better_context_kind_apc = kwargs.get("better_context_kind_apc", None)
            y = self.encoder.module.compute_better_context(y, kind=better_context_kind_apc)

        out = self.classifier(y)  # (batch * 1)
        return out

    def _propagate_ev(self, data_loader: DataLoader, optimizer, status: str, **kwargs):
        """
        :param status: train / validate / test
        """
        device = self.device_ids[0]

        loss, step = 0., 0
        for subx_view1, subx_view2, label in data_loader:  # subx: (batch * n_channels * subseq_len)
            optimizer.zero_grad()
            label = self._adjust_label(label)

            out = self._out_from_encoder_classifier(subx_view1.to(device), **kwargs)

            # loss
            loss = self.criterion(out, label.to(device))

            # weight update
            if status == "train":
                loss.backward()
                optimizer.step()
                self.lr_scheduler.step()

            loss += loss.item()
            step += 1

        return loss / step

    @torch.no_grad()
    def _validate_ev(self, optimizer, **kwargs):
        val_loss = self._propagate_ev(self.val_data_loader, optimizer, "validate", **kwargs)
        return val_loss

    @torch.no_grad()
    def _test_ev(self, optimizer, **kwargs):
        test_loss = self._propagate_ev(self.test_data_loader, optimizer, "test", **kwargs)
        return test_loss

    @staticmethod
    def _enable_bn_stat_update_only(model):
        """
        Set `training=True` for BatchNorm(s).
        """
        for child in get_children(model):
            child.training = True if isinstance(child, nn.BatchNorm1d) or isinstance(child, nn.BatchNorm2d) or isinstance(child, IterNorm) else False

    @abstractmethod
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

    @torch.no_grad()
    def _compute_test_acc(self, **kwargs):
        device = self.device_ids[0]

        test_acc, step = 0., 0
        for subx_view1, subx_view2, label in self.test_data_loader:  # subx: (batch * n_channels * subseq_len)
            label = self._adjust_label(label)

            out = self._out_from_encoder_classifier(subx_view1.to(device), **kwargs)

            # compute `test_acc`
            out = torch.argmax(out, dim=1).detach().cpu()
            test_acc += accuracy_score(out, label)

            step += 1

        return test_acc / step

    def _compute_test_macroAUC(self, subseq_len, **kwargs):
        """
        defined in `evaluator_ptbxl.py`
        """
        pass

    def fit(self, optimizer, n_epochs_ev: dict, **kwargs):
        wb_logger = WBLogger() if self.use_wandb else None

        for epoch in range(1, n_epochs_ev[self.evaluation_type]):
            self.epoch = epoch

            # loss
            self._train_eval_setting_during_fit("train")
            train_loss = self._propagate_ev(self.train_data_loader, optimizer, "train", **kwargs)

            # validate & test
            self._train_eval_setting_during_fit("validate")
            val_loss = self._validate_ev(optimizer, **kwargs)
            self._train_eval_setting_during_fit("test")
            test_loss = self._test_ev(optimizer, **kwargs)

            # evaluation (test accuracy | test macro AUC)
            test_acc, test_macroAUC = None, None
            if self.dataset_name == "PTB-XL":
                test_macroAUC = self._compute_test_macroAUC(**kwargs)
            else:
                test_acc = self._compute_test_acc(**kwargs)

            # status log
            if self.use_wandb:
                wb_logger.main_log(epoch, train_loss, val_loss, test_loss, test_acc=test_acc, test_macroAUC=test_macroAUC)
                wb_logger.main_summary(train_loss, val_loss, test_loss, test_acc=test_acc, test_macroAUC=test_macroAUC)
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                wb_logger.log(epoch=epoch, lr_clf=lrs[0], lr_enc=lrs[-1])

        self.finish_wandb()
