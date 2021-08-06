import numpy as np
import wandb

import torch
import torch.nn as nn
from torch import relu
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

from vibcreg.normalization.iter_norm import IterNorm
from vibcreg.lr_scheduler.cosine_annealing_lr import CosineAnnealingLR

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
        self.count = 0

    @staticmethod
    def log(**kwargs):
        log_content = {k: v for k, v in kwargs.items()}
        wandb.log(log_content)

    @staticmethod
    def main_log(epoch, train_loss, val_loss, test_loss, test_acc):
        keys = ['epoch', 'train_loss', 'val_loss', 'test_loss', 'test_acc']
        vals = [epoch, train_loss, val_loss, test_loss, test_acc]
        log_content = {k: v for k, v in zip(keys, vals)}
        wandb.log(log_content)

    def main_summary(self, train_loss, val_loss, test_loss, test_acc):
        """
        Note that `wand.run.summary` must be run at every step along with `wand.log(..)`.
        """
        if self.best_train_loss is None:
            self.best_train_loss = train_loss
            self.best_val_loss = val_loss
            self.best_test_loss = test_loss
            self.best_test_acc = test_acc

        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss

        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc

        wandb.run.summary['train_loss'] = self.best_train_loss
        wandb.run.summary['val_loss'] = self.best_val_loss
        wandb.run.summary['test_loss'] = self.best_test_loss
        wandb.run.summary['test_acc'] = self.best_test_acc

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


class Evaluator(object):
    """
    It is used for both 'linear evaluation' and 'fine-tuning evaluation'.
    """
    def __init__(self, cf, train_data_loader, val_data_loader, test_data_loader):
        self.cf = cf
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

        self.criterion = nn.CrossEntropyLoss()

        # params
        self.fine_tune = self.set_fine_tune(cf["evaluation_type"])
        self.encoder = None
        self.rl_model = None
        self.ar_cpc = None
        self.classifier = None
        self.wb = None
        self.lr_scheduler = None

    @staticmethod
    def set_fine_tune(evaluation_type: str) -> bool:
        if evaluation_type == "linear_evaluation":
            fine_tune = False
        elif evaluation_type == "fine_tuning_evaluation":
            fine_tune = True
        else:
            raise ValueError("invalid `evaluation_type`")
        return fine_tune

    @staticmethod
    def _freeze(model):
        for param in model.parameters():
            param.requires_grad = False

    def load_model(self, encoder, framework_type, loading_checkpoint_fname, device_ids, **kwargs):
        # load
        if framework_type == "supervised":
            self.encoder = encoder
            self.encoder = nn.DataParallel(self.encoder, device_ids=device_ids)
        else:
            if framework_type == "rand_init":
                self.rl_model = RandInit(encoder)
            elif (framework_type == "vibcreg") or (framework_type == "vbibcreg"):
                self.rl_model = VIbCReg(encoder, encoder.last_channels_enc, **self.cf)
            elif framework_type == "barlow_twins":
                self.rl_model = BarlowTwins(encoder, encoder.last_channels_enc, **self.cf)
            elif framework_type == "simsiam":
                self.rl_model = SimSiam(encoder, encoder.last_channels_enc, **self.cf)
            elif framework_type == "cpc":
                self.rl_model = CPC(encoder, **self.cf)
                self.ar_cpc = nn.DataParallel(self.rl_model.ar, device_ids=device_ids)
            elif framework_type == "apc":
                self.rl_model = APC(encoder, **self.cf)
            else:
                raise ValueError("invalid `framework_type`")

            checkpoint = torch.load(loading_checkpoint_fname)
            self.rl_model.load_state_dict(checkpoint['model_state_dict'])
            self.encoder = self.rl_model.encoder
            self.encoder = nn.DataParallel(self.encoder, device_ids=device_ids)
            self._freeze(self.encoder) if not self.fine_tune else None

    def build_classifier(self, framework_type, device_ids, **kwargs):
        # compute `in_size` and `out_size` for the classifier
        if framework_type == 'cpc':
            in_size = self.rl_model.module.ar.ar.input_size
        elif framework_type == "apc":
            better_context_kind_apc = kwargs.get("better_context_kind_apc", None)
            mul = len(better_context_kind_apc.split("+"))
            in_size = self.encoder.module.rnn.hidden_size * mul
        else:
            # if `encoder` is `ResNet1D`.
            in_size = self.encoder.module.res_blocks[-1].conv_1x1.out_channels
        out_size = len(np.unique(self.train_data_loader.dataset.Y))

        # build a classifier
        self.classifier = nn.Sequential(nn.Linear(in_size, out_size))
        self.classifier = nn.DataParallel(self.classifier, device_ids=device_ids)

    def init_wandb(self, dataset_name, framework_type, loading_checkpoint_fname, project_name, use_wandb, **kwargs):
        # set `run_name`
        run_name = f"{framework_type}"
        if dataset_name == "UCR":
            ucr_dataset_name = kwargs.get("ucr_dataset_name", None)
            run_name = f"{ucr_dataset_name}-" + run_name

        if loading_checkpoint_fname:
            ep = int(loading_checkpoint_fname.split("ep_")[1].split(".pth")[0])
            run_name = run_name + f"-ep_{ep}"

        if self.fine_tune:
            train_data_ratio = kwargs.get("train_data_ratio", None) / 8 * 1000  # 8: 8 training folds; 1000 to make it percentage.
            run_name = run_name + f"-{train_data_ratio}"

        # set `project_name`
        if not self.fine_tune:
            project_name = project_name + "-LE"  # linear evaluation
        else:
            project_name = project_name + "-FtE"  # fine-tuning evaluation

        # initialize wandb
        if use_wandb:
            self.wb = wandb.init(project=project_name, config=self.cf, name=run_name)
        else:
            self.wb = None

    def wandb_watch(self, use_wandb, **kwargs):
        if not use_wandb:
            return None
        wandb.watch(self.encoder)
        wandb.watch(self.classifier)

    def finish_wandb(self, use_wandb, **kwargs):
        self.wb.finish() if use_wandb else None

    def setup_lr_scheduler(self, optimizer, kind: str, device_ids: list, batch_size: int, n_epochs_ev: dict, evaluation_type: str, **kwargs):
        if kind == "CosineAnnealingLR":
            train_dataset_size = kwargs.get("train_dataset_size", None)
            n_gpus = len(device_ids)
            self.lr_scheduler = CosineAnnealingLR(optimizer, train_dataset_size, n_gpus, batch_size, n_epochs_ev[evaluation_type]).get_lr_scheduler()
        else:
            raise ValueError("unavailable name for `lr_scheduler`.")

    def trainable_params(self, framework_type: str, lr_clf_ev: float, lr_enc_ev: float, **kwargs):
        """
        returns 'trainable parameters'
        """
        if (framework_type == "supervised") or (self.fine_tune is True):
            params = [{"params": self.classifier.parameters(), "lr": lr_clf_ev},
                      {"params": self.encoder.parameters(), "lr": lr_enc_ev}]
        else:  # linear evaluation
            params = [{"params": self.classifier.parameters(), "lr": lr_enc_ev}]
        return params

    @staticmethod
    def adjust_label(label):
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

    def _propagate_ev(self, data_loader: DataLoader, optimizer, status: str, device_ids: list, **kwargs):
        """
        :param status: train / validate / test
        """
        device = device_ids[0]

        loss, step = 0., 0
        for subx_view1, subx_view2, label in data_loader:  # subx: (batch * n_channels * subseq_len)
            optimizer.zero_grad()
            label = self.adjust_label(label)

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

    @torch.no_grad()
    def _compute_test_acc(self, device_ids: list, **kwargs):
        device = device_ids[0]

        test_acc, step = 0., 0
        for subx_view1, subx_view2, label in self.test_data_loader:  # subx: (batch * n_channels * subseq_len)
            label = self.adjust_label(label)

            out = self._out_from_encoder_classifier(subx_view1.to(device), **kwargs)

            # compute `test_acc`
            out = torch.argmax(out, dim=1).detach().cpu()
            test_acc += accuracy_score(out, label)

            step += 1

        return test_acc / step

    @staticmethod
    def _enable_bn_stat_update_only(model):
        """
        Set `training=True` for BatchNorm only, while setting `training=False` for Dropout.
        """
        for child in get_children(model):
            child.training = True if isinstance(child, nn.BatchNorm1d) or isinstance(child, nn.BatchNorm2d) or isinstance(child, IterNorm) else False

    def fit(self, optimizer, n_epochs_ev: dict, evaluation_type: str, framework_type: str, use_wandb: bool, **kwargs):
        wb_logger = WBLogger() if use_wandb else None

        for epoch in range(1, n_epochs_ev[evaluation_type]):

            # encoder
            if (framework_type == 'supervised') or (framework_type == "apc"):
                self.encoder.train()
            elif evaluation_type == "fine_tuning_evaluation":
                self._enable_bn_stat_update_only(self.encoder)
            else:
                self.encoder.eval()

            # classifier
            self.classifier.train()

            # loss
            train_loss = self._propagate_ev(self.train_data_loader, optimizer, "train", **kwargs)

            # validate
            self.encoder.eval()
            self.classifier.eval()
            if framework_type == "apc":
                self.encoder.train()
            val_loss = self._validate_ev(optimizer, **kwargs)
            test_loss = self._test_ev(optimizer, **kwargs)

            # test accuracy
            test_acc = self._compute_test_acc(**kwargs)

            # status log
            if use_wandb:
                wb_logger.main_log(epoch, train_loss, val_loss, test_loss, test_acc)
                wb_logger.main_summary(train_loss, val_loss, test_loss, test_acc)
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                wb_logger.log(epoch=epoch, lr_clf=lrs[0], lr_enc=lrs[-1])

        self.finish_wandb(use_wandb)