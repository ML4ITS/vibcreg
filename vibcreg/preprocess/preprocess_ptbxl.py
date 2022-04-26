import ast

import numpy as np
import pandas as pd
import wfdb
from torch.utils.data import Dataset

from vibcreg.preprocess.augmentations import Augmentations
from vibcreg.util import get_git_root


class LabelEncoder(object):
    def __init__(self):
        self.data_root = get_git_root().joinpath("vibcreg", "data", "PTB-XL")
        df = pd.read_csv(self.data_root.joinpath("scp_statements.csv"))
        self.abb2idx = {}
        i = 0
        for abb in df.iloc[:, 0]:
            self.abb2idx[abb] = i
            i += 1

    def encode(self, scp_codes: list):
        """
        scp_codes: a list of abbreviations.

        it receives `scp_codes` and returns a multi-label as in [https://machinelearningmastery.com/multi-label-classification-with-deep-learning/].
        """
        label = np.zeros(len(self.abb2idx))
        for scp_code in scp_codes:
            label[self.abb2idx[scp_code]] = 1.
        return label


class PTB_XL(Dataset):
    def __init__(self, kind: str, augs: Augmentations, used_augmentations: list,
                 train_fold=(1, 2, 3, 4, 5, 6, 7, 8), val_fold=9, test_fold=10,
                 train_downsampling_rate=1.,
                 sampling_rate=100,
                 is_tnc_used: bool = False,
                 return_rc_dist: bool = False,
                 **kwargs):
        """
        :param kind: "train" / "validate" / "test"
        :param augs: instance of the `Augmentations` class.
        :param used_augmentations: e.g., ["RC", "AmpR", "Vshift"]
        :param train_fold: 1~8
        :param val_fold: 9
        :param test_fold: 10
        :param train_downsampling_rate: it's used for the fine-tuning evaluation. {1: no-downsampling; 0.1: use 10% of a training dataset}
        :param sampling_rate: It's set to 100Hz in the related papers.
        """
        super(PTB_XL, self).__init__()
        self.kind = kind
        self.augs = augs
        self.used_augmentations = used_augmentations
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.train_downsampling_rate = train_downsampling_rate
        self.sampling_rate = sampling_rate
        self.is_tnc_used = is_tnc_used
        self.return_rc_dist = return_rc_dist
        self.data_root = get_git_root().joinpath("vibcreg", "data", "PTB-XL")

        # load and process annotation data
        Y = self.load_annotation_data()
        self.Y_ = self.split_annotation_data_into_train_val_test(Y)

        # params
        self.label_encoder = LabelEncoder()
        self._len = self.Y_.shape[0]

    def load_annotation_data(self):
        # Y = pd.read_csv("./data/PTB-XL/ptbxl_database.csv", index_col='ecg_id')
        Y = pd.read_csv(self.data_root.joinpath("ptbxl_database.csv"), index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        return Y

    def split_annotation_data_into_train_val_test(self, Y):
        """
        split the annotation data into train, valid, test datasets.
        """
        if self.kind == 'train':
            train_idices = Y.strat_fold.apply(lambda x: x in self.train_fold)
            for i in range(len(train_idices)):
                if train_idices.iloc[i]:
                    if np.random.rand() > self.train_downsampling_rate:
                        train_idices.iloc[i] = False
            Y_ = Y[train_idices]
        elif self.kind == 'validate':
            val_idices = Y.strat_fold.apply(lambda x: x == self.val_fold)
            Y_ = Y[val_idices]
        elif self.kind == 'test':
            test_idices = Y.strat_fold.apply(lambda x: x == self.test_fold)
            Y_ = Y[test_idices]
        else:
            raise ValueError(f'invalid `kind`: {self.kind}')
        return Y_

    def _get_a_sample(self, idx):
        """
        get a sample of (`x`: signal, `y`: annotation)
        """
        y = self.Y_.iloc[idx]
        if self.sampling_rate == 100:
            fname = y.filename_lr  # lr: low rate: 100Hz
        elif self.sampling_rate == 500:  # unavailable in this code; we need the `records500` data dir.
            fname = y.filename_hr  # hr: high rate: 500Hz
        else:
            raise ValueError("invalid `sampling_rate`.")
        x, _ = wfdb.rdsamp(str(self.data_root.joinpath(fname)))  # x: (seq_len * 12)
        x = np.transpose(x, (1, 0))  # (12 * seq_len)
        return x, y

    def _create_label(self, y):
        """
        creates a label according to a trainable format.
        """
        scp_codes = list(y['scp_codes'].keys())
        label = self.label_encoder.encode(scp_codes)
        return label

    @staticmethod
    def _assign_float32(*xs):
        """
        assigns `dtype` of `float32`
        so that we wouldn't have to change `dtype` later before propagating data through a model.
        """
        new_xs = []
        for x in xs:
            new_xs.append(x.astype(np.float32))
        return new_xs[0] if (len(xs) == 1) else new_xs

    def getitem_default(self, idx):
        # get a sample (`x`: signal, `y`: annotation)
        x, y = self._get_a_sample(idx)  # x: (12 * seq_len)
        std_x = np.std(x, axis=1, keepdims=True)  # (12, 1)
        label = self._create_label(y)  # (71,)

        subx_view1, subx_view2 = x.copy(), x.copy()

        # augmentations
        used_augs = [] if self.kind == "test" else self.used_augmentations
        for aug in used_augs:
            if aug == "RC":  # random crop
                subx_view1, subx_view2 = self.augs.random_crop(subx_view1, subx_view2)
            if aug == "AmpR":  # random amplitude resize
                subx_view1, subx_view2 = self.augs.amplitude_resize(subx_view1, subx_view2)
            if aug == "Vshift":  # random vertical shift
                subx_view1, subx_view2 = self.augs.vertical_shift(std_x, subx_view1, subx_view2)

        subx_view1, subx_view2 = self._assign_float32(subx_view1, subx_view2)

        if self.return_rc_dist:
            return subx_view1, subx_view2, label, np.array([self.augs.rc_dist]).astype(float)
        else:
            return subx_view1, subx_view2, label

    def getitem_tnc(self, idx):
        # get a sample (`x`: signal, `y`: annotation)
        x, y = self._get_a_sample(idx)  # x: (12 * seq_len)
        std_x = np.std(x, axis=1, keepdims=True)  # (12, 1)
        label = self._create_label(y)  # (71,)

        subx_view1, subx_view2, subx_view3 = self.augs.neigh_random_crop(x)

        # augmentations
        used_augs = [] if self.kind == "test" else self.used_augmentations
        for aug in used_augs:
            if aug == "AmpR":  # random amplitude resize
                subx_view1, subx_view2, subx_view3 = self.augs.amplitude_resize(subx_view1, subx_view2, subx_view3)
            if aug == "Vshift":  # random vertical shift
                subx_view1, subx_view2, subx_view3 = self.augs.vertical_shift(std_x, subx_view1, subx_view2, subx_view3)

        subx_view1, subx_view2, subx_view3 = self._assign_float32(subx_view1, subx_view2, subx_view3)
        return subx_view1, subx_view2, subx_view3, label

    def __getitem__(self, idx):
        if self.is_tnc_used:
            return self.getitem_tnc(idx)
        else:
            return self.getitem_default(idx)

    def __len__(self):
        return self._len


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    # os.chdir("../")

    # data pipeline
    batch_size = 6
    augs = Augmentations(subseq_len=250)  # 2.5s * 100Hz
    train_dataset = PTB_XL("train", augs, ["RC", "AmpR", "Vshift"], train_downsampling_rate=1.)  # ["RC", "AmpR", "Vshift"]
    test_dataset = PTB_XL("test", augs, [])
    train_data_loader = DataLoader(train_dataset, batch_size, num_workers=0, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size, num_workers=0, shuffle=True)

    # get a mini-batch of samples
    for subx_view1, subx_view2, y in train_data_loader:
    # for subx_view1, subx_view2, y in test_data_loader:
        break
    print("subx_view1.shape:", subx_view1.shape)
    print("y.shape:", y.shape)

    # plot
    # 1. multiple w.r.t `batch_idx`
    batch_indices = np.arange(batch_size)
    lead_idx = 0
    plt.figure(figsize=(10, 1.2 * 3))
    for i, batch_idx in enumerate(batch_indices):
        plt.subplot(3, 2, i + 1)
        for j, signal in enumerate([subx_view1, subx_view2]):
            plt.title(f'batch_idx: {batch_idx}')
            plt.plot(signal[batch_idx, lead_idx, :], '-' if j == 0 else '-', linewidth=1, alpha=0.5 if j != 0 else 1.)
    plt.tight_layout()
    plt.show()
    # 2. multiple w.r.t `lead_idx`
    batch_idx = 1
    lead_idices = np.arange(12)
    plt.figure(figsize=(10, 1.2 * 6))
    for i, lead_idx in enumerate(lead_idices):
        plt.subplot(6, 2, i + 1)
        for j, signal in enumerate([subx_view1, subx_view2]):
            plt.title(f'lead_idx: {lead_idx}')
            plt.plot(signal[batch_idx, lead_idx, :], '-' if j == 0 else '-', linewidth=1, alpha=0.5 if j != 0 else 1.)
    plt.tight_layout()
    plt.show()
