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


class Evaluator(object):
    """
    It is used for both 'linear evaluation' and 'fine-tuning evaluation'.
    """
    def __init__(self, cf, train_data_loader, val_data_loader, test_data_loader):
        self.cf = cf
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

        # params
        self.fine_tune = self._set_fine_tune(cf["evaluation_type"])
        self.criterion = self._set_criterion(cf["dataset_name"])
        self.encoder = None
        self.rl_model = None
        self.ar_cpc = None
        self.classifier = None
        self.wb = None
        self.lr_scheduler = None
        self.epoch = None

    @staticmethod
    def _set_fine_tune(evaluation_type: str) -> bool:
        if evaluation_type == "linear_evaluation":
            fine_tune = False
        elif evaluation_type == "fine_tuning_evaluation":
            fine_tune = True
        else:
            raise ValueError("invalid `evaluation_type`")
        return fine_tune

    @staticmethod
    def _set_criterion(dataset_name: str):
        if dataset_name == "UCR":
            criterion = nn.CrossEntropyLoss()
        elif dataset_name == "PTB-XL":
            criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Define `self.criterion` for your `dataset_name`.")
        return criterion

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

    def _get_n_unique_labels(self, dataset_name):
        if dataset_name == "UCR":
            n_unique_labels = len(np.unique(self.train_data_loader.dataset.Y))
        elif dataset_name == "PTB-XL":
            n_unique_labels = 71
        else:
            raise ValueError('define `n_unique_labels` for your `dataset_name`.')
        return n_unique_labels

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
        out_size = self._get_n_unique_labels(kwargs.get("dataset_name"))

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

        if self.fine_tune and kwargs.get("train_data_ratio", None):
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
            if kwargs.get("dataset_name") == "UCR":
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

    @torch.no_grad()
    def _compute_test_macroAUC(self, framework_type, subseq_len, device_ids, **kwargs):
        """
        References:
        [1] Scikit-learn, "Plot ROC curves for the multilabel problem" (https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py)
        """
        device = device_ids[0]

        # Collect the entire test data
        labels = torch.tensor([])  # (entire_batch * 71)
        cls_preds = torch.tensor([])  # (entire_batch * 71)
        for subx_view1, subx_view2, label in self.test_data_loader:  # subx: (batch * n_channels * subseq_len)
            labels = torch.cat((labels, label))

            if framework_type == 'cpc':
                z = self.encoder(subx_view1.float().to(device))  # (batch * n_channels * reduced_seq_len)
                out = z.mean(dim=2)  # (batch * n_channels)
                out = self.classifier(out).cpu().detach()  # (batch * 71)
                minibatch_cls_pred = torch.sigmoid(out)
            else:
                # averaged classification prediction
                i = 0
                minibatch_cls_preds = torch.tensor([])  # (n_slides, batch * 71)
                while subx_view1[:, :, i:i + subseq_len].shape[-1] == subseq_len:
                    subx = subx_view1[:, :, i:i + subseq_len].float()  # (batch * 12 * subseq_len)
                    z = self.encoder(subx.to(device))  # (batch * feature_size)
                    out = self.classifier(z).cpu().detach()  # (batch * 71)
                    out = torch.sigmoid(out)  # (batch * 71)
                    out = out.unsqueeze(dim=0)  # (1 * batch * 71)
                    minibatch_cls_preds = torch.cat((minibatch_cls_preds, out))  # (n_slides * batch * 71)
                    i += subseq_len
                minibatch_cls_pred = minibatch_cls_preds.mean(dim=0)  # (batch * 71)
            cls_preds = torch.cat((cls_preds, minibatch_cls_pred))  # (entire_batch * 71)

        # Compute Macro-AUC
        # - compute ROC curve and ROC area for each class
        fpr = dict()  # false positive rate
        tpr = dict()  # true positive rate
        roc_auc = dict()
        n_classes = labels.shape[-1]
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(labels[:, i], cls_preds[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # - first aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # - then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # - finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # confusion matrix
        self._log_confusion_matrix(labels, cls_preds, **kwargs)
        return roc_auc["macro"]

    def _log_confusion_matrix(self, y_test, y_pred, use_wandb, tsne_analysis_log_epochs, **kwargs):
        """
        [1] Stackoverflow, https://stackoverflow.com/questions/62722416/plot-confusion-matrix-for-multilabel-classifcation-python
        """
        if not use_wandb:
            return None
        if self.epoch not in tsne_analysis_log_epochs:
            return None

        y_pred = np.round(y_pred)  # i.e., threshold = 0.5
        n_classes = y_test.shape[-1]

        y_test = np.abs(1 - y_test)  # to make 0 False and 1 True.
        y_pred = np.abs(1 - y_pred)

        f, axes = plt.subplots(round(n_classes / 8), 8, figsize=(25, 15))
        axes = axes.ravel()
        for i in range(n_classes):
            disp = ConfusionMatrixDisplay(confusion_matrix(y_test[:, i], y_pred[:, i]), display_labels=[0, i])
            disp.plot(ax=axes[i], values_format='.4g')
            disp.ax_.set_title(f'class {i}')
            if i < 10:
                disp.ax_.set_xlabel('')
            if i % 5 != 0:
                disp.ax_.set_ylabel('')
            disp.im_.colorbar.remove()
            disp.ax_.set_xticks([])
            disp.ax_.set_yticks([])

        plt.subplots_adjust(wspace=0.10, hspace=0.5)
        f.colorbar(disp.im_, ax=axes)
        plt.suptitle(f"epoch: {self.epoch}")
        wandb.log({f'confusion_mat-ep_{self.epoch}': plt})
        plt.close()

    @staticmethod
    def _enable_bn_stat_update_only(model):
        """
        Set `training=True` for BatchNorm only, while setting `training=False` for Dropout.
        """
        for child in get_children(model):
            child.training = True if isinstance(child, nn.BatchNorm1d) or isinstance(child, nn.BatchNorm2d) or isinstance(child, IterNorm) else False

    def fit(self, optimizer, n_epochs_ev: dict, evaluation_type: str, **kwargs):
        dataset_name = kwargs.get("dataset_name", None)
        framework_type = kwargs.get("framework_type", None)
        use_wandb = kwargs.get("use_wandb", None)
        wb_logger = WBLogger() if use_wandb else None

        for epoch in range(1, n_epochs_ev[evaluation_type]):
            self.epoch = epoch

            # encoder trainable setting
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

            # evaluation (test accuracy | test macro AUC)
            test_acc, test_macroAUC = None, None
            if dataset_name == "UCR":
                test_acc = self._compute_test_acc(**kwargs)
            elif dataset_name == "PTB-XL":
                test_macroAUC = self._compute_test_macroAUC(**kwargs)
            else:
                test_acc = self._compute_test_acc(**kwargs)

            # status log
            if use_wandb:
                wb_logger.main_log(epoch, train_loss, val_loss, test_loss, test_acc=test_acc, test_macroAUC=test_macroAUC)
                wb_logger.main_summary(train_loss, val_loss, test_loss, test_acc=test_acc, test_macroAUC=test_macroAUC)
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                wb_logger.log(epoch=epoch, lr_clf=lrs[0], lr_enc=lrs[-1])

        self.finish_wandb(use_wandb)
