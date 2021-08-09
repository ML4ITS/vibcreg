"""
Mean ShiFt (MSF)
- Unlike the original paper, it'd be used as an "add-in" for other frameworks by adding its weighted loss.

[1] S. Koohpayegani et al., 2021, "Mean Shift for Self-Supervised Learning"
[2] MSF original GitHub: https://github.com/UMBCvision/MSF
"""
import copy

import torch
import torch.nn as nn


class MemoryBank(object):
    """
    based on MoCo's memory bank.
    First-in-First-out system.
    """
    def __init__(self, size_mb, k_msf, device_ids, feature_size_msf, batch_size):
        self.size_mb = size_mb  # memory bank size; 4096 is used in the MoCo paper.
        self.k = k_msf  # a number of nearest neighbors used; 5 is recommended in the MSF paper.
        self.feature_size = feature_size_msf
        self.batch_size = batch_size

        self.bank = self._initialize_memory_bank().to(device_ids[0])  # (size_mb * feature_size)
        self.ptr = 0

    def _initialize_memory_bank(self):
        assert self.size_mb % self.batch_size == 0  # if condition returns True, then nothing happens:
        bank = torch.randn(self.size_mb, self.feature_size)  # (size_mb * feature_size)
        bank = nn.functional.normalize(bank, dim=-1)
        return bank

    def enqueue_dequeue(self, z_target):
        """
        :param z_target: (batch_size * feature_size); (= `u` in the MSF paper.)
        - this `z_target` is obtained by a target encoder.
        """
        batch_size = z_target.shape[0]
        if batch_size == self.batch_size:
            self.bank[self.ptr: self.ptr + batch_size] = z_target  # enqueue and dequeue
            self.ptr = (self.ptr + batch_size) % self.size_mb  # move pointer
        else:
            pass


class AddinMSF(object):
    def __init__(self, memory_bank: MemoryBank, tau_msf: float = 0.99, use_EMAN_msf: bool = False):
        self.memory_bank = memory_bank
        self.tau = tau_msf  # 0.99 is used in the MSF paper.
        self.use_EMAN = use_EMAN_msf

        self.target_net = None

    def create_target_net(self, online_net):
        """
        initialized as a copy of an online network.
        """
        self.target_net = copy.deepcopy(online_net)

    def update_target_net(self, online_net):
        """
        `target_net` is updated by moving average of the online network.

        - updated at each training "step"
        [1] https://github.com/lucidrains/byol-pytorch/blob/2aa84ee18fafecaf35637da4657f92619e83876d/byol_pytorch/byol_pytorch.py#L61
        """
        for online_param, target_param in zip(online_net.parameters(), self.target_net.parameters()):
            target_param.data = self.tau * target_param.data + (1 - self.tau) * online_param.data

        if self.use_EMAN:
            for online_buffer, target_buffer in zip(online_net.buffers(), self.target_net.buffers()):
                target_buffer.data = self.tau * target_buffer.data + (1 - self.tau) * online_buffer.data

    def compute_loss(self, z_target, z_online):
        """
        :param z_target: (batch_size * feature_size); (= `u` in the MSF paper.)
        :param z_online: (batch_size * feature_size); (= `v` in the MSF paper.)

        - `keys` are the representations by the target encoder stored in the memory bank.
        - `dist` "cosine similarity" used in the MSF paper.
        - we can't use Euclidean distance and MSELoss due to the computational limit and slowness.
        - the loss by MSF would be weighted and added to an original loss of a used framework.

        [1] https://github.com/danelee2601/MSF/blob/main/train_msf.py
        """
        # z_target = z_target.cpu().detach()  # it gets stored in the memory bank.
        # z_online = z_online.cpu()
        z_target = z_target.detach()  # it gets stored in the memory bank.

        z_target = nn.functional.normalize(z_target, dim=-1)
        z_online = nn.functional.normalize(z_online, dim=-1)

        self.memory_bank.enqueue_dequeue(z_target)

        # Calculate mean shift regression loss (cosine similarity in a MSELoss form)
        # - calculate distances between vectors
        # - torch.einsum: [1] https://ita9naiwa.github.io/numeric%20calculation/2018/11/10/Einsum.html
        dist_t = 2 - 2 * torch.einsum('bc,kc->bk', [z_target, self.memory_bank.bank])  # queue = targets; current_target = z_target
        dist_q = 2 - 2 * torch.einsum('bc,kc->bk', [z_online, self.memory_bank.bank])  # query = z_online

        # select the k nearest neighbors [with smallest distance (largest=False)] based on current target
        _, nn_index = dist_t.topk(self.memory_bank.k, dim=1, largest=False)
        nn_dist_q = torch.gather(dist_q, 1, nn_index)

        loss = (nn_dist_q.sum(dim=1) / self.memory_bank.k).mean()

        return loss
