"""
`Dataset` (pytorch) class is defined.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from vibcreg.util import get_git_root
from vibcreg.data.download_data import download_ucr_datasets
from vibcreg.preprocess.augmentations import Augmentations


class DatasetImporter(object):
    """
    1. It imports one dataset from the UCR archive.
    2. splits into `X_train`, `X_test`, `Y_train`, and `Y_test`.
    """
    def __init__(self, ucr_dataset_name: str,
                 train_data_ratio: int = 0.8,
                 test_data_ratio: int = 0.2,
                 train_random_seed: int = 0,
                 test_random_seed: int = 0, **kwargs):
        """
        :param ucr_dataset_name: e.g., "ElectricDevices"
        :param train_data_ratio: 0.8 means 80% of a dataset
        :param test_data_ratio: 0.2 means 20% of a dataset
        :param train_random_seed:
        :param test_random_seed:
        """
        download_ucr_datasets()
        self.data_root = get_git_root().joinpath("vibcreg", "data", "UCRArchive_2018", ucr_dataset_name)

        # fetch an entire dataset
        df_train = pd.read_csv(self.data_root.joinpath(f"{ucr_dataset_name}_TRAIN.tsv"), sep='\t', header=None)
        df_test = pd.read_csv(self.data_root.joinpath(f"{ucr_dataset_name}_TEST.tsv"), sep='\t', header=None)
        df = pd.concat((df_train, df_test), axis=0)
        X, Y = df.iloc[:, 1:].values, df.iloc[:, [0]].values

        rand_indices = np.arange(X.shape[0])
        rand_indices_train, rand_indices_test, sub_Y, _ = train_test_split(rand_indices, Y,
                                                                           train_size=train_data_ratio,
                                                                           test_size=test_data_ratio,
                                                                           stratify=Y.reshape(-1),
                                                                           random_state=test_random_seed)

        # sub_rand_indices, rand_indices_test, sub_Y, _ = train_test_split(rand_indices, Y,
        #                                                                  test_size=test_data_ratio,
        #                                                                  stratify=Y.reshape(-1),
        #                                                                  random_state=test_random_seed)
        # test_size = 1 - (train_data_ratio * (1 / (1 - test_data_ratio)))
        #
        # if test_size == 0:
        #     rand_indices_train = sub_rand_indices.copy()
        # else:
        #     rand_indices_train, _, _, _ = train_test_split(sub_rand_indices, sub_Y,
        #                                                    test_size=test_size,
        #                                                    stratify=sub_Y.reshape(-1),
        #                                                    random_state=train_random_seed)

        self.X_train, self.X_test = X[rand_indices_train, :], X[rand_indices_test, :]  # (B x F)
        self.Y_train, self.Y_test = Y[rand_indices_train, :], Y[rand_indices_test, :]  # (B x 1)

        print("# rand_indices_train.shape:", rand_indices_train.shape)
        print("# rand_indices_test.shape:", rand_indices_test.shape)

        print("# unique labels (train):", np.unique(self.Y_train.reshape(-1)))
        print("# unique labels (test):", np.unique(self.Y_test.reshape(-1)))


class UCRDataset(Dataset):
    def __init__(self,
                 kind: str,
                 dataset_importer: DatasetImporter,
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
        x = np.arcsinh(x)
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
        x, y = self.X[idx, :], self.Y[idx, :]
        x = x.reshape(1, -1)  # (1 x F)
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
        x, y = self.X[idx, :], self.Y[idx, :]
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

    # data pipeline
    augs = Augmentations(subseq_len=48)
    dataset_importer = DatasetImporter("ElectricDevices")
    train_dataset = UCRDataset("train", dataset_importer, augs, ["RC", "AmpR", "Vshift"],
                               is_tnc_used=False, return_rc_dist=True)
    # test_dataset = UCRDataset("test", dataset_importer, augs, [])
    train_data_loader = DataLoader(train_dataset, batch_size=32, num_workers=0, shuffle=True)
    # test_data_loader = DataLoader(test_dataset, batch_size=32, num_workers=0, shuffle=True)

    # get a mini-batch of samples
    for batch in train_data_loader:
        # subx_view1, subx_view2, y = batch
        # subx_view1, subx_view2, subx_view3, y = batch
        subx_view1, subx_view2, y, rc_dist = batch
        break

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
