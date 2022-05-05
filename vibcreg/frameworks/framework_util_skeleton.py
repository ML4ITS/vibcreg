"""
It provides a skeleton for the `Utility` classes of each SSL framework.
There is no skeleton for a SSL framework since `torch.nn.Module` already provides it.
"""
import os
from abc import ABC, abstractmethod

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from vibcreg.lr_scheduler.cosine_annealing_lr import CosineAnnealingLR
from vibcreg.util import get_git_root


class Utility_SSL(ABC):
    @abstractmethod
    def __init__(self, rl_model, device_ids: list, framework_type="vibcreg",
                 use_wandb=True, run_name=None, **kwargs):
        """
        :param rl_model: instance of a SSL model
        :param device_ids: a list of gpu-device-ids.
        :param framework_type:
        :param use_wandb:
        :param run_name: a run name in the W&B project. If None, it's automatically set.
        """
        self.rl_model = rl_model
        self.device_ids = device_ids
        self.device = device_ids[0]
        self.framework_type = framework_type
        self.use_wandb = use_wandb
        self.run_name = run_name

        # params
        self.epoch = None
        self.global_step = 0
        self.lr_scheduler = None
        self.wb = None
        self.reprs = None
        self.labels = None
        self.vibcreg_folder = get_git_root().joinpath("vibcreg")

    def update_epoch(self, epoch):
        self.epoch = epoch

    def setup_lr_scheduler(self, optimizer, batch_size, n_epochs, kind="CosineAnnealingLR", **kwargs):
        if kind == "CosineAnnealingLR":
            train_dataset_size = kwargs.get("train_dataset_size", None)
            n_gpus = len(self.device_ids)
            self.lr_scheduler = CosineAnnealingLR(optimizer, train_dataset_size, n_gpus, batch_size, n_epochs).get_lr_scheduler()
        else:
            raise ValueError("unavailable name for `lr_scheduler`.")

    def init_wandb(self, config_dataset, config_framework, overwritten_project_name=None):
        matplotlib.use('Agg')  # eliminates the issue of 'TclError: Can't find a usable tk.tcl in the following directories:' when using `matplotlib`.

        # set `run_name`
        if config_dataset["dataset_name"] == "UCR":
            ucr_dataset_name = config_dataset["ucr_dataset_name"]
            run_name = f"{ucr_dataset_name}-{self.framework_type}"
        elif config_dataset["dataset_name"] == "UEA":
            uea_dataset_name = config_dataset["uea_dataset_name"]
            run_name = f"{uea_dataset_name}-{self.framework_type}"
        else:
            run_name = f"{self.framework_type}"

        if self.run_name is not None:
            run_name = self.run_name

        # initialize wandb
        if self.use_wandb:
            config = {}
            for cf in [config_dataset, config_framework]:
                for k, v in cf.items():
                    config[k] = v

            if overwritten_project_name is None:
                project_name = config_framework["project_name"].get(config_dataset["dataset_name"], None)
                if project_name is None:
                    project_name = config_dataset["dataset_name"]
            else:
                project_name = overwritten_project_name

            self.wb = wandb.init(project=project_name, config=config, name=run_name)
        else:
            self.wb = None

    def finish_wandb(self):
        if self.use_wandb:
            self.wb.finish()

    @staticmethod
    def _compute_feature_decorr_metrics(z):
        """
        z: (batch * h_size)
        To see the feature decorrelation magnitude.
        `redundancy_metrics` ranges from -1 (negatively-correlated) to one (positively-correlated); `zero` denotes de-correlated.
        """
        z = z - z.mean(dim=0)
        z = F.normalize(z, p=2, dim=0)  # batch-wise l2-normalize
        z = z.detach().cpu().numpy()  # (batch * feature)

        cov = np.matmul(z.T, z)
        np.fill_diagonal(cov, 0.)
        feature_decorr_metrics = np.mean(np.abs(cov))
        return feature_decorr_metrics

    @staticmethod
    def _feature_comp_expressiveness_metrics(z):
        """
        originally from the SimSiam paper introduced as `output_std`.
        `output_std` denotes a standard deviation in one feature component in a batch.
        Here, we name it as 'feature component expressiveness metrics'
        """
        z = z.detach()
        feature_comp_expr = torch.std(z, dim=0).mean().item()
        return feature_comp_expr

    @abstractmethod
    def wandb_watch(self):
        if self.use_wandb:
            wandb.watch(...)

    def status_log_per_iter(self, status, z, **kwargs):
        """
        :param status: "train" / "validate"
        :param z: representation `z`
        :param kwargs:
        """
        if self.use_wandb and (status == "train"):
            wandb.log({'global_step': self.global_step, 'feature_comp_expr_metrics': self._feature_comp_expressiveness_metrics(z), 'feature_decorr_metrics': self._compute_feature_decorr_metrics(z)})
        elif self.use_wandb and (status == "validate"):
            pass

    @abstractmethod
    def representation_learning(self, data_loader, optimizer, status):
        """
        :param status: train / validate / test
        """
        self.rl_model.train() if status == "train" else self.rl_model.eval()

        loss, step = 0., 0
        for subx_view1, subx_view2, label in data_loader:  # subx: (batch * n_channels * subseq_len)
            optimizer.zero_grad()

            # loss
            L = ...

            # weight update
            if status == "train":
                L.backward()
                optimizer.step()
                self.lr_scheduler.step()
                self.global_step += 1

            loss += L.item()
            step += 1

            # status log
            self.status_log_per_iter(status, ...)  # 2nd argument: representation

        return loss / step


    @torch.no_grad()
    def validate(self, val_data_loader, optimizer, dataset_name: str, n_neighbors_kNN: int, n_jobs_for_kNN: int, **kwargs):
        self.rl_model.eval()
        val_loss = self.representation_learning(val_data_loader, optimizer, "validate")

        if self.use_wandb:
            if dataset_name == "PTB-XL":
                self.log_macro_f1score_during_validation(val_data_loader, n_neighbors_kNN, n_jobs_for_kNN)
            else:
                self.log_kNN_acc_during_validation(val_data_loader, n_neighbors_kNN, n_jobs_for_kNN)

        return val_loss

    @torch.no_grad()
    def test(self, test_data_loader, optimizer):
        test_loss = self.representation_learning(test_data_loader, optimizer, "test")
        return test_loss

    # @staticmethod
    def _stack_val_data(self, val_data_loader):
        subxs = torch.tensor([])
        labels = torch.tensor([])
        for batch in val_data_loader:  # subx: (batch * n_channels * subseq_len)
            if (self.framework_type == 'tnc') or (self.framework_type == 'vnibcreg'):
                subx_view1, subx_view2, subx_view3, label = batch
            elif self.framework_type == 'vibcreg_rcd':
                subx_view1, subx_view2, label, rc_dist = batch
            else:
                subx_view1, subx_view2, label = batch
            subxs = torch.cat((subxs, subx_view1), dim=0)
            labels = torch.cat((labels, label), dim=0)
        return subxs, labels

    @abstractmethod
    def _representation_for_validation(self, x):
        """
        :param x: input data

        `representation` computed here is used for
        - [UCR] the kNN accuracy during validation.
        - [PTB-XL] macro F1 score during validation.
        """
        repr_ = self.rl_model.module.encoder(x.to(self.device)).detach().cpu()
        return repr_

    @torch.no_grad()
    def log_kNN_acc_during_validation(self, val_data_loader, n_neighbors_kNN, n_jobs_for_kNN):
        """
        Log kNN-accuracy on a validation dataset.
        """
        subxs, labels = self._stack_val_data(val_data_loader)
        repr_ = self._representation_for_validation(subxs)
        labels = labels.view(-1)

        model = KNeighborsClassifier(n_neighbors_kNN, n_jobs=n_jobs_for_kNN)
        model.fit(repr_, labels)
        pred_labels = model.predict(repr_)
        acc = accuracy_score(labels, pred_labels)
        wandb.log({'epoch': self.epoch, 'kNN_acc': acc})

    @torch.no_grad()
    def log_macro_f1score_during_validation(self, val_data_loader, n_neighbors_kNN, n_jobs_for_kNN, min_n_labels=10):
        """
        computes kNN-macro-F1-score on the validation dataset.
        In computing the f1-score, classes that have less than `min_n_labels` are not considered.
        """
        subxs, labels = self._stack_val_data(val_data_loader)
        repr_ = self._representation_for_validation(subxs)

        model = KNeighborsClassifier(n_neighbors_kNN)
        multi_target_model = MultiOutputClassifier(model, n_jobs=n_jobs_for_kNN)
        multi_target_model.fit(repr_, labels)
        pred_labels = multi_target_model.predict(repr_)
        n_classes = labels.shape[-1]
        metric = 0.
        count = 0
        for i in range(n_classes):
            if (labels[:, i]).sum() >= min_n_labels:  # consider `labels[:,i]` that contains at least one `True` only.
                metric += f1_score(labels[:, i], pred_labels[:, i], average='macro', zero_division=1)
                count += 1
        metric /= count
        wandb.log({'epoch': self.epoch, 'macro_f1_score': metric})

    def save_checkpoint(self, epoch, optimizer, train_loss, val_loss, model_saving_epochs=(10, 100)):

        if epoch in model_saving_epochs:
            filename = f"checkpoint-{self.framework_type}-ep_{epoch}.pth"
            savepath = self.vibcreg_folder.joinpath("checkpoints", filename)

            try:
                if not os.path.isdir(self.vibcreg_folder.joinpath("checkpoints")):
                    os.mkdir(self.vibcreg_folder.joinpath("checkpoints"))

                torch.save({'epoch': epoch,
                            'model_state_dict': self.rl_model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            }, savepath)
            except PermissionError:
                print("model couldn't be saved due to some permission error.")
                pass

    @torch.no_grad()
    def get_batch_of_representations(self, test_data_loader, dataset_name: str, **kwargs):
        self.rl_model.module.encoder.eval()

        if dataset_name == "PTB-XL":
            NORM_idx = test_data_loader.dataset.label_encoder.abb2idx['NORM']
            SR_idx = test_data_loader.dataset.label_encoder.abb2idx['SR']
            reprs = torch.tensor([])
            isNORMs = torch.tensor([])
            for batch in test_data_loader:  # subx: (batch * 12 * subseq_len); label: (batch * 71); 71 unique classes.
                if (self.framework_type == 'tnc') or (self.framework_type == 'vnibcreg'):
                    subx_view1, subx_view2, subx_view3, label = batch
                elif self.framework_type == 'vibcreg_rcd':
                    subx_view1, subx_view2, label, rc_dist = batch
                else:
                    subx_view1, subx_view2, label = batch
                repr_ = self._representation_for_validation(subx_view1)
                reprs = torch.cat((reprs, repr_), dim=0)

                target = torch.zeros(71)  # 71 sub-classes
                target[NORM_idx], target[SR_idx] = 1, 1
                isNORM = ((label == target).float().mean(dim=1) == 1).float()
                isNORMs = torch.cat((isNORMs, isNORM))
            self.reprs = reprs.numpy()
            self.labels = isNORMs.numpy()  # classes are divided as 1) abnormal(class:0), 2) normal(class:1)
        else:
            reprs = torch.tensor([])
            labels = torch.tensor([])
            for batch in test_data_loader:  # subx: (batch * 12 * subseq_len); label: (batch * 71); 71 unique classes.
                if (self.framework_type == 'tnc') or (self.framework_type == 'vnibcreg'):
                    subx_view1, subx_view2, subx_view3, label = batch
                elif self.framework_type == 'vibcreg_rcd':
                    subx_view1, subx_view2, label, rc_dist = batch
                else:
                    subx_view1, subx_view2, label = batch
                repr_ = self._representation_for_validation(subx_view1)
                reprs = torch.cat((reprs, repr_), dim=0)
                labels = torch.cat((labels, label.cpu()), dim=0)
            self.reprs = reprs.numpy()
            self.labels = labels.numpy().reshape(-1)

    def log_feature_histogram(self, n_samples=1000):
        """
        It logs a plot of histograms in W&B;
        - Histogram of n-th values in the representation vector w.r.t a mini-batch.
        - to see whether the representations have collapsed.
        """
        if not self.use_wandb:
            return None

        f, a = plt.subplots(6, 6, figsize=(12, 12))
        a = a.ravel()
        for feature_idx, ax in enumerate(a):
            ax.set_title(f'{feature_idx + 1}-th')
            ax.hist(self.reprs[:n_samples, feature_idx], bins=100)
        plt.tight_layout()
        # wandb.log({f"Feature histogram-ep_{self.epoch}": [wandb.Image(f, caption=f'')]})
        wandb.log({f"Feature histogram": [wandb.Image(f, caption=f'')]})
        plt.close()
        print("# log_feature_histogram")

    def log_tsne_analysis(self, n_samples=-1):
        if not self.use_wandb:
            return None

        # run t-SNE
        y_embedded = TSNE(n_components=2, random_state=1).fit_transform(self.reprs[:n_samples, :])
        plt.figure(figsize=(5, 5))
        plt.scatter(y_embedded[:, 0], y_embedded[:, 1], alpha=0.5, c=self.labels[:n_samples], cmap="nipy_spectral")
        plt.tight_layout()
        # wandb.log({f't-SNE analysis-ep_{self.epoch}': [wandb.Image(plt, caption=f'')]})
        wandb.log({f't-SNE analysis': [wandb.Image(plt, caption=f'')]})
        plt.close()
        print("# log_tsne_analysis")

    @staticmethod
    def print_train_status(epoch, lr, train_loss, val_loss, print_epoch_interval=1):
        if epoch % print_epoch_interval == 0:
            print(f"epoch: {epoch}, lr: {round(lr, 5)}, train_loss: {round(train_loss, 5)}, val_loss: {round(val_loss, 5)}")

    def status_log(self, epoch, lr, train_loss, val_loss):
        """Log the running results in W&B"""
        if self.use_wandb:
            wandb.log({'epoch': epoch, 'lr': lr, 'train_loss': train_loss, 'val_loss': val_loss})

    def test_log(self, test_loss):
        if self.use_wandb:
            wandb.log({'test_loss': test_loss})
