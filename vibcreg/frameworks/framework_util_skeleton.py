"""
It provides a skeleton for the `Utility` classes of each SSL framework.
There is no skeleton for a SSL framework since `torch.nn.Module` already provides it.
"""
from abc import ABC, abstractmethod

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import wandb

import torch
import torch.nn.functional as F

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score

from vibcreg.lr_scheduler.cosine_annealing_lr import CosineAnnealingLR


class Utility_SSL(ABC):
    @abstractmethod
    def __init__(self, rl_model, device_ids: list, batch_size=256, n_epochs=100, framework_type="vibcreg", weight_on_msfLoss=0.,
                 use_wandb=True, project_name="RLonUCR", run_name=None, n_neighbors_kNN=5, n_jobs_for_kNN=10, model_saving_epochs=(10, 100), **kwargs):
        """
        :param rl_model: instance of a SSL model
        :param device_ids: a list of gpu-device-ids.
        :param batch_size:
        :param n_epochs:
        :param framework_type:
        :param weight_on_msfLoss:
        :param use_wandb:
        :param project_name: a project name in W&B.
        :param run_name: a run name in the W&B project. If None, it's automatically set.
        :param n_neighbors_kNN: n_neighbors for kNN.
        :param n_jobs_for_kNN: n_cpus for kNN.
        :param model_saving_epochs: a list of epochs when the model is saved.
        """
        self.rl_model = rl_model
        self.device_ids = device_ids
        self.device = device_ids[0]
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.framework_type = framework_type
        self.weight_on_msfLoss = weight_on_msfLoss
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.run_name = run_name
        self.n_neighbors_kNN = n_neighbors_kNN
        self.n_jobs_for_kNN = n_jobs_for_kNN
        self.model_saving_epochs = model_saving_epochs

        # params
        self.epoch = None
        self.global_step = 0
        self.lr_scheduler = None
        self.wb = None
        self.ys = None
        self.labels = None

    def update_epoch(self, epoch):
        self.epoch = epoch

    def setup_lr_scheduler(self, optimizer, kind="CosineAnnealingLR", **kwargs):
        if kind == "CosineAnnealingLR":
            train_dataset_size = kwargs.get("train_dataset_size", None)
            n_gpus = len(self.device_ids)
            self.lr_scheduler = CosineAnnealingLR(optimizer, train_dataset_size, n_gpus, self.batch_size, self.n_epochs).get_lr_scheduler()
        else:
            raise ValueError("unavailable name for `lr_scheduler`.")

    def init_wandb(self, cf):
        dataset_name = cf["dataset_name"]
        matplotlib.use('Agg')  # eliminates the issue of 'TclError: Can't find a usable tk.tcl in the following directories:' when using `matplotlib`.

        # set `run_name`
        if dataset_name == "UCR":
            ucr_dataset_name = cf["ucr_dataset_name"]
            run_name = f"{ucr_dataset_name}-{self.framework_type}"
        else:
            run_name = f"{self.framework_type}"

        if self.run_name is not None:
            run_name = self.run_name

        # initialize wandb
        if self.use_wandb:
            self.wb = wandb.init(project=self.project_name, config=cf, name=run_name)
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
        pass

    @abstractmethod
    def representation_learning(self, data_loader, optimizer, status):
        """
        :param status: train / validate / test
        """
        pass

    @torch.no_grad()
    def validate(self, val_data_loader, optimizer, dataset_name, **kwargs):
        val_loss = self.representation_learning(val_data_loader, optimizer, "validate")

        if self.use_wandb:
            if dataset_name == "UCR":
                self.log_kNN_acc_during_validation(val_data_loader)
            elif dataset_name == "PTB-XL":
                self.log_macro_f1score_during_validation(val_data_loader)
            else:
                raise ValueError("invalid `dataset_name`.")

        return val_loss

    @torch.no_grad()
    def test(self, test_data_loader, optimizer):
        test_loss = self.representation_learning(test_data_loader, optimizer, "test")
        return test_loss

    @torch.no_grad()
    def log_kNN_acc_during_validation(self, val_data_loader):
        """
        Log kNN-accuracy on a validation dataset.
        """
        self.rl_model.eval()

        subxs = torch.tensor([])
        labels = torch.tensor([])
        for subx_view1, subx_view2, label in val_data_loader:  # subx: (batch * n_channels * subseq_len)
            subxs = torch.cat((subxs, subx_view1), dim=0)
            labels = torch.cat((labels, label), dim=0)

        y = self.rl_model.module.encoder(subxs.to(self.device)).detach().cpu()
        labels = labels.view(-1)

        model = KNeighborsClassifier(self.n_neighbors_kNN, n_jobs=self.n_jobs_for_kNN)
        model.fit(y, labels)
        pred_labels = model.predict(y)
        acc = accuracy_score(labels, pred_labels)
        wandb.log({'epoch': self.epoch, 'kNN_acc': acc})

    @torch.no_grad()
    def log_macro_f1score_during_validation(self, val_data_loader, min_n_labels=10):
        """
        computes kNN-macro-F1-score on the validation dataset.
        In computing the f1-score, classes that have less than `min_n_labels` are not considered.
        """
        self.rl_model.eval()

        subxs = torch.tensor([])
        labels = torch.tensor([])
        for subx_view1, subx_view2, label in val_data_loader:  # subx: (batch * n_channels * subseq_len)
            subxs = torch.cat((subxs, subx_view1), dim=0)
            labels = torch.cat((labels, label), dim=0)

        y = self.rl_model.module.encoder(subxs.to(self.device)).detach().cpu()

        model = KNeighborsClassifier(self.n_neighbors_kNN)
        multi_target_model = MultiOutputClassifier(model, n_jobs=self.n_jobs_for_kNN)
        multi_target_model.fit(y, labels)
        pred_labels = multi_target_model.predict(y)
        n_classes = labels.shape[-1]
        metric = 0.
        count = 0
        for i in range(n_classes):
            if (labels[:, i]).sum() >= min_n_labels:  # consider `labels[:,i]` that contains at least one `True` only.
                metric += f1_score(labels[:, i], pred_labels[:, i], average='macro', zero_division=1)
                count += 1
        metric /= count
        wandb.log({'epoch': self.epoch, 'macro_f1_score': metric})

    def save_checkpoint(self, epoch, optimizer, train_loss, val_loss):
        model_saving_path = "./checkpoints/frameworks/checkpoint-{}-ep_{}.pth"

        if epoch in self.model_saving_epochs:
            torch.save({'epoch': epoch,
                        'model_state_dict': self.rl_model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        }, model_saving_path.format(self.framework_type, epoch))

    @torch.no_grad()
    def get_batch_of_representations(self, test_data_loader, dataset_name, **kwargs):
        encoder = self.rl_model.module.encoder
        encoder.eval()

        if dataset_name == "UCR":
            ys = torch.tensor([])
            labels = torch.tensor([])
            for subx_view1, subx_view2, label in test_data_loader:  # subx: (batch * 1 * subseq_len)
                y = encoder(subx_view1.to(self.device)).to('cpu')  # (batch_size * feature_size)
                ys = torch.cat((ys, y), dim=0)
                labels = torch.cat((labels, label.to('cpu')), dim=0)
            self.ys = ys.numpy()
            self.labels = labels.numpy().reshape(-1)

        elif dataset_name == "PTB-XL":
            NORM_idx = test_data_loader.dataset.label_encoder.abb2idx['NORM']
            SR_idx = test_data_loader.dataset.label_encoder.abb2idx['SR']
            ys = torch.tensor([])
            isNORMs = torch.tensor([])
            for subx_view1, subx_view2, label in test_data_loader:  # subx: (batch * 12 * subseq_len); label: (batch * 71); 71 unique classes.
                y = encoder(subx_view1.to(self.device))  # (batch_size * feature_size)
                ys = torch.cat((ys, y.to('cpu')), dim=0)

                target = torch.zeros(71)  # 71 sub-classes
                target[NORM_idx], target[SR_idx] = 1, 1
                isNORM = ((label == target).float().mean(dim=1) == 1).float()
                isNORMs = torch.cat((isNORMs, isNORM))
            self.ys = ys.numpy()
            self.labels = isNORMs.numpy()  # classes are divided as 1) abnormal(class:0), 2) normal(class:1)

        else:
            raise ValueError("invalid `dataset_name`.")

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
            ax.hist(self.ys[:n_samples, feature_idx], bins=100)
        plt.tight_layout()
        wandb.log({f"Feature histogram-ep_{self.epoch}": [wandb.Image(f, caption=f'')]})
        plt.close()
        print("# log_feature_histogram")

    def log_tsne_analysis(self, n_samples=-1):
        if not self.use_wandb:
            return None

        # run t-SNE
        y_embedded = TSNE(n_components=2, random_state=1).fit_transform(self.ys[:n_samples, :])
        plt.figure(figsize=(5, 5))
        plt.scatter(y_embedded[:, 0], y_embedded[:, 1], alpha=0.5, c=self.labels[:n_samples], cmap="nipy_spectral")
        plt.tight_layout()
        wandb.log({f't-SNE analysis-ep_{self.epoch}': [wandb.Image(plt, caption=f'')]})
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