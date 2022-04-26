from abc import ABC, abstractmethod

import numpy as np
import torch


class LRScheduler(ABC):
    @abstractmethod
    def get_lr_scheduler(self):
        pass


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, train_dataset_size, n_gpus, batch_size, n_epochs, T_max=None, eta_min=0.):
        """
        :param optimizer:
        :param train_dataset_size:
        :param n_gpus:
        :param batch_size:
        :param n_epochs:
        :param T_max: Maximum number of iterations.
        :param eta_min: Minimum learning rate.
        """
        self.optimizer = optimizer
        self.train_dataset_size = train_dataset_size
        self.n_gpus = n_gpus
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.T_max = T_max
        self.eta_min = eta_min

    def _compute_n_train_iters_in_a_batch(self):
        """
        i.e., a number of mini-batches of a training dataset.
        """
        n_train_iters_in_a_batch = np.ceil(self.train_dataset_size / self.batch_size)
        return n_train_iters_in_a_batch

    def get_lr_scheduler(self):
        n_train_iters_in_a_batch = self._compute_n_train_iters_in_a_batch()
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                  T_max=int(n_train_iters_in_a_batch * self.n_epochs + n_train_iters_in_a_batch) // self.n_gpus if not self.T_max else self.T_max,
                                                                  eta_min=self.eta_min)
        return lr_scheduler
