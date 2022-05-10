"""
`Dataset` (pytorch) class is defined.
"""
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sktime.datasets import load_from_tsfile
from sklearn.preprocessing import LabelEncoder

from vibcreg.util import get_git_root
from vibcreg.data.download_data import download_ucr_datasets
from vibcreg.preprocess.augmentations import Augmentations


class DatasetImporterDefaultUEA(object):
    """
    This uses train and test sets as given.
    To compare with the results from ["Unsupervised scalable representation learning for multivariate time series"]
    """

    def __init__(self, uea_dataset_name: str, **kwargs):
        """
        :param uea_dataset_name: e.g., "ElectricDevices"
        :param train_data_ratio: 0.8 means 80% of a dataset
        :param test_data_ratio: 0.2 means 20% of a dataset
        :param train_random_seed:
        :param test_random_seed:
        """
        download_ucr_datasets()
        self.data_root = get_git_root().joinpath("vibcreg", "data", "UEAArchive_2018", uea_dataset_name)

        # fetch an entire dataset
        self.X_train, self.Y_train = load_from_tsfile(str(self.data_root.joinpath(f'{uea_dataset_name}_TRAIN.ts')),
                                                      return_data_type='numpy3d', )
        self.X_test, self.Y_test = load_from_tsfile(str(self.data_root.joinpath(f'{uea_dataset_name}_TEST.ts')),
                                                    return_data_type='numpy3d')

        le = LabelEncoder()
        self.Y_train = le.fit_transform(self.Y_train)
        self.Y_test = le.transform(self.Y_test)
        # self.Y_train, self.Y_test = np.array(self.Y_train, dtype=float).astype(int), np.array(self.Y_test, dtype=float).astype(int)

        print('self.X_train.shape:', self.X_train.shape)
        print('self.X_test.shape:', self.X_test.shape)

        print("# unique labels (train):", np.unique(self.Y_train.reshape(-1)))
        print("# unique labels (test):", np.unique(self.Y_test.reshape(-1)))


class UEADataset(Dataset):
    def __init__(self,
                 kind: str,
                 dataset_importer,
                 augs: Augmentations,
                 used_augmentations: list,
                 data_scaling: bool = True,
                 is_tnc_used: bool = False,
                 return_rc_dist: bool = False,
                 **kwargs):
        """
        :param kind: "train" / "test"
        :param dataset_importer: instance of the `DatasetImporter` class.
        :param augs: instance of the `Augmentations` class.
        :param used_augmentations: e.g., ["RC", "AmpR", "Vshift"]
        :param data_scaling: whether to scale input data.
        """
        super().__init__()
        self.kind = kind
        self.augs = augs
        self.used_augmentations = used_augmentations if kind == "train" else []
        self.data_scaling = data_scaling
        self.is_tnc_used = is_tnc_used
        self.return_rc_dist = return_rc_dist

        if kind == "train":
            self.X, self.Y = dataset_importer.X_train, dataset_importer.Y_train
        elif kind == "test":
            self.X, self.Y = dataset_importer.X_test, dataset_importer.Y_test
        else:
            raise ValueError

        self._len = self.X.shape[0]

    @staticmethod
    def _scale_data(x):
        """
        :param x: input time series data (n_channels x sequence_length)
        :return: scaled `x`

        Scaling: z-normalization + arcsinh
        """
        x = x - np.mean(x)
        x = x / np.std(x)
        # x = np.arcsinh(x)
        return x

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
        x, y = self.X[idx, :, :], self.Y[idx]

        if len(x.shape) == 1:
            x = x[np.newaxis, :]  # to make a channel dim of 1 for a univariate time series
        std_x = np.std(x)

        # scale time series (whole sequence)
        if self.data_scaling:
            x = self._scale_data(x)

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
            return subx_view1, subx_view2, y, np.array([self.augs.rc_dist]).astype(float)
        else:
            return subx_view1, subx_view2, y

    def getitem_tnc(self, idx):
        x, y = self.X[idx, :, :], self.Y[idx]
        x = x.reshape(1, -1)  # (1 x F)
        std_x = np.std(x)

        # scale time series (whole sequence)
        if self.data_scaling:
            x = self._scale_data(x)

        # subx_view1, subx_view2 = x.copy(), x.copy()
        subx_view1, subx_view2, subx_view3 = self.augs.neigh_random_crop(x)

        # augmentations
        used_augs = [] if self.kind == "test" else self.used_augmentations
        for aug in used_augs:
            if aug == "AmpR":  # random amplitude resize
                subx_view1, subx_view2, subx_view3 = self.augs.amplitude_resize(subx_view1, subx_view2, subx_view3)
            if aug == "Vshift":  # random vertical shift
                subx_view1, subx_view2, subx_view3 = self.augs.vertical_shift(std_x, subx_view1, subx_view2, subx_view3)

        subx_view1, subx_view2, subx_view3 = self._assign_float32(subx_view1, subx_view2, subx_view3)
        return subx_view1, subx_view2, subx_view3, y

    def __getitem__(self, idx):
        if self.is_tnc_used:
            return self.getitem_tnc(idx)
        else:
            return self.getitem_default(idx)

    def __len__(self):
        return self._len


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    os.chdir("../")

    uea_dataset_name = 'SpokenArabicDigits'
    data_root = get_git_root().joinpath("vibcreg", "data", "UEAArchive_2018", uea_dataset_name)
    test_x, test_y = load_from_tsfile(str(data_root.joinpath(f'{uea_dataset_name}_TEST.ts')))
    seq_len = len(test_x.iloc[0, 0])
    print('seq_len:', seq_len)

    # data pipeline
    augs = Augmentations(subseq_len=48)
    dataset_importer = DatasetImporterDefaultUEA(uea_dataset_name)
    # train_dataset = UEADataset("train", dataset_importer, augs, ["RC", "AmpR"])
    train_dataset = UEADataset("train", dataset_importer, augs, ['AmpR'])
    # test_dataset = UCRDataset("test", dataset_importer, augs, [])
    train_data_loader = DataLoader(train_dataset, batch_size=32, num_workers=0, shuffle=True)
    # test_data_loader = DataLoader(test_dataset, batch_size=32, num_workers=0, shuffle=True)

    # get a mini-batch of samples
    for batch in train_data_loader:
        subx_view1, subx_view2, y = batch
        break

    print('subx_view1.shape:', subx_view1.shape)

    # plot
    plt.figure(figsize=(9, 4))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.plot(subx_view1[i, 0, :])
        plt.plot(subx_view2[i, 0, :])
        # if len(batch) == 4:
        #     plt.plot(subx_view3[i, 0, :])
        plt.grid()
    plt.show()

# if __name__ == "__main__":
#     import os
#     import matplotlib.pyplot as plt
#     from torch.utils.data import DataLoader
#     os.chdir("../")
#
#     # data pipeline
#     augs = Augmentations(subseq_len=48)
#     dataset_importer = DatasetImporterDefault("ElectricDevices")
#     test_dataset = UCRDataset("test", dataset_importer, augs, [])
#     test_data_loader = DataLoader(test_dataset, batch_size=32, num_workers=0, shuffle=True)
#
#     # get a mini-batch of samples
#     for batch in test_data_loader:
#         subx_view1, _, y = batch
#         break
#     print(subx_view1)
#
#     # plot
#     plt.figure(figsize=(9, 4))
#     for i in range(9):
#         plt.subplot(3, 3, i + 1)
#         plt.plot(subx_view1[i, 0, :])
#         plt.grid()
#     plt.show()
