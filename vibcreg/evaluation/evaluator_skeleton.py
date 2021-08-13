from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from vibcreg.lr_scheduler.cosine_annealing_lr import CosineAnnealingLR
from vibcreg.normalization.iter_norm import IterNorm
from vibcreg.wrapper.model_building_wrapper import ModelBuilder


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
    def __init__(self, config_dataset, config_framework, config_eval,
                 train_data_loader, val_data_loader, test_data_loader,
                 evaluation_type, loading_checkpoint_fname, device_ids, use_wandb,
                 **kwargs):
        self.config_dataset = config_dataset
        self.config_framework = config_framework
        self.config_eval = config_eval
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.evaluation_type = evaluation_type
        self.loading_checkpoint_fname = loading_checkpoint_fname
        self.device_ids = device_ids
        self.device = device_ids[0]
        self.use_wandb = use_wandb

        # params
        self.framework_type = self.config_framework.get("framework_type", None)
        self.dataset_name = self.config_dataset.get("dataset_name", None)
        self.fine_tune = self._set_fine_tune()
        self.criterion = self._set_criterion()

        self.encoder = None
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

    def load_encoder(self) -> None:
        """
        load `self.encoder` internally.
        """
        if self.framework_type == "supervised":
            config_framework_ = self.config_framework.copy()
            config_framework_["backbone_type"] = "resnet1d"
            model_builder = ModelBuilder(self.config_dataset, config_framework_, self.device_ids, self.use_wandb)
            self.encoder = model_builder.build_encoder()
            self.encoder = nn.DataParallel(self.encoder, device_ids=self.device_ids)
        else:
            # build model (encoder + SSL framework)
            model_builder = ModelBuilder(self.config_dataset, self.config_framework, self.device_ids, self.use_wandb)
            encoder = model_builder.build_encoder()
            rl_model, _ = model_builder.build_model(encoder, apply_data_parallel=False)

            # load
            checkpoint = torch.load(self.loading_checkpoint_fname)
            rl_model.load_state_dict(checkpoint['model_state_dict'])
            self.encoder = nn.DataParallel(rl_model.encoder, device_ids=self.device_ids)
            self._freeze(self.encoder) if not self.fine_tune else None

    @abstractmethod
    def _clf_in_size(self) -> int:
        """
        get an input size of a classification head.
        """
        if self.framework_type == 'cpc':
            in_size = self.config_framework.get("enc_hid_channels_cpc", None)
        elif self.framework_type == "apc":
            better_context_kind_apc = self.config_framework.get("better_context_kind_apc", None)
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

    def build_classifier(self) -> None:
        """
        build `self.classifier` internally.
        """
        in_size = self._clf_in_size()
        out_size = self._clf_out_size()

        # build a classifier
        self.classifier = nn.Sequential(nn.Linear(in_size, out_size))
        self.classifier = nn.DataParallel(self.classifier, device_ids=self.device_ids)

    def init_wandb(self):
        # set `run_name`
        run_name = f"{self.framework_type}"
        if self.dataset_name == "UCR":
            ucr_dataset_name = self.config_dataset.get("ucr_dataset_name", None)
            run_name = f"{ucr_dataset_name}-" + run_name

        if self.loading_checkpoint_fname:
            ep = int(self.loading_checkpoint_fname.split("ep_")[1].split(".pth")[0])
            run_name = run_name + f"-ep_{ep}"

        if self.fine_tune and (self.dataset_name == "UCR"):
            train_data_ratio = self.config_dataset.get("train_data_ratio", None) / 8 * 1000  # 8: 8 training folds; 1000 to make it percentage.
            run_name = run_name + f"-{train_data_ratio}"

        # set `project_name`
        project_name = self.config_framework["project_name"].get(self.dataset_name, "")
        if not self.fine_tune:
            project_name += "-LE"  # linear evaluation
        else:
            project_name += "-FtE"  # fine-tuning evaluation

        # initialize wandb
        if self.use_wandb:
            # combine configs
            config = {}
            for cf in [self.config_dataset, self.config_framework, self.config_eval]:
                for k, v in cf.items():
                    config[k] = v
            self.wb = wandb.init(project=project_name, config=config, name=run_name)
        else:
            self.wb = None

    def wandb_watch(self):
        if not self.use_wandb:
            return None
        wandb.watch(self.encoder)
        wandb.watch(self.classifier)

    def finish_wandb(self):
        self.wb.finish() if self.use_wandb else None

    def setup_lr_scheduler(self, optimizer, kind: str, **kwargs):
        batch_size = self.config_dataset["batch_size"]
        n_epochs_ev = self.config_eval["n_epochs_ev"][self.evaluation_type]
        if kind == "CosineAnnealingLR":
            train_dataset_size = kwargs.get("train_dataset_size", None)
            n_gpus = len(self.device_ids)
            self.lr_scheduler = CosineAnnealingLR(optimizer, train_dataset_size, n_gpus, batch_size, n_epochs_ev).get_lr_scheduler()
        else:
            raise ValueError("unavailable name for `lr_scheduler`.")

    def trainable_params(self) -> list:
        """
        returns 'trainable parameters'
        """
        if (self.framework_type == "supervised") or (self.fine_tune is True):
            params = [{"params": self.classifier.parameters(), "lr": self.config_eval["lr_clf_ev"]},
                      {"params": self.encoder.parameters(), "lr": self.config_eval["lr_enc_ev"]}]
        else:  # linear evaluation
            params = [{"params": self.classifier.parameters(), "lr": self.config_eval["lr_clf_ev"]}]
        return params

    @abstractmethod
    def _adjust_label(self, label):
        """
        receives `label` and processes it if necessary, and returns it.
        """
        pass

    def _out_from_encoder_classifier(self, x):
        """
        returns output from an encoder-classifier.
        """
        y = self.encoder(x)
        if self.framework_type == "apc":
            better_context_kind_apc = self.config_framework.get("better_context_kind_apc", None)
            y = self.encoder.module.compute_better_context(y, kind=better_context_kind_apc)

        out = self.classifier(y)  # (batch * 1)
        return out

    def _propagate_ev(self, data_loader: DataLoader, optimizer, status: str):
        """
        :param status: train / validate / test
        """
        loss, step = 0., 0
        for subx_view1, subx_view2, label in data_loader:  # subx: (batch * n_channels * subseq_len)
            optimizer.zero_grad()
            label = self._adjust_label(label)

            out = self._out_from_encoder_classifier(subx_view1.to(self.device))

            # loss
            loss = self.criterion(out, label.to(self.device))

            # weight update
            if status == "train":
                loss.backward()
                optimizer.step()
                self.lr_scheduler.step()

            loss += loss.item()
            step += 1

        return loss / step

    @torch.no_grad()
    def _validate_ev(self, optimizer):
        val_loss = self._propagate_ev(self.val_data_loader, optimizer, "validate")
        return val_loss

    @torch.no_grad()
    def _test_ev(self, optimizer):
        test_loss = self._propagate_ev(self.test_data_loader, optimizer, "test")
        return test_loss

    @staticmethod
    def _enable_bn_stat_update_only(model):
        """
        Set `training=True` for BatchNorm(s).
        """
        for child in get_children(model):
            child.training = True if isinstance(child, nn.BatchNorm1d) or isinstance(child, nn.BatchNorm2d) or isinstance(child, IterNorm) else False

    @abstractmethod
    def _train_eval_setting_during_fit(self, status: str) -> None:
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
                self.encoder.train()  # throws an error otherwise
            self.classifier.eval()

    @torch.no_grad()
    def _compute_test_acc(self):
        test_acc, step = 0., 0
        for subx_view1, subx_view2, label in self.test_data_loader:  # subx: (batch * n_channels * subseq_len)
            label = self._adjust_label(label)

            out = self._out_from_encoder_classifier(subx_view1.to(self.device))

            # compute `test_acc`
            out = torch.argmax(out, dim=1).detach().cpu()
            test_acc += accuracy_score(out, label)

            step += 1

        return test_acc / step

    def _compute_test_macroAUC(self, subseq_len: int):
        """
        defined in `evaluator_ptbxl.py`
        """
        pass

    def fit(self, optimizer):
        n_epochs_ev = self.config_eval["n_epochs_ev"][self.evaluation_type]
        wb_logger = WBLogger() if self.use_wandb else None

        for epoch in range(1, n_epochs_ev):
            self.epoch = epoch

            # loss
            self._train_eval_setting_during_fit("train")
            train_loss = self._propagate_ev(self.train_data_loader, optimizer, "train")

            # validate & test
            self._train_eval_setting_during_fit("validate")
            val_loss = self._validate_ev(optimizer)
            self._train_eval_setting_during_fit("test")
            test_loss = self._test_ev(optimizer)

            # evaluation (test accuracy | test macro AUC)
            test_acc, test_macroAUC = None, None
            if self.dataset_name == "PTB-XL":
                test_macroAUC = self._compute_test_macroAUC(self.config_dataset["subseq_len"])
            else:
                test_acc = self._compute_test_acc()

            # status log
            if self.use_wandb:
                wb_logger.main_log(epoch, train_loss, val_loss, test_loss, test_acc=test_acc, test_macroAUC=test_macroAUC)
                wb_logger.main_summary(train_loss, val_loss, test_loss, test_acc=test_acc, test_macroAUC=test_macroAUC)
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                wb_logger.log(epoch=epoch, lr_clf=lrs[0], lr_enc=lrs[-1])

        self.finish_wandb()
