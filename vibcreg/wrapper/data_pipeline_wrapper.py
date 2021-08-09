import yaml
from torch.utils.data import DataLoader

from vibcreg.preprocess.augmentations import Augmentations
from vibcreg.preprocess.preprocess_ucr import DatasetImporter, UCRDataset
from vibcreg.preprocess.preprocess_ptbxl import PTB_XL


def load_hyper_param_settings(yaml_fname: str):
    """
    :param yaml_fname: .yaml file that consists of hyper-parameter settings.

    For example,
    [UCR] `yaml_fname`: "./examples/configs/example_ucr_vibcreg.yaml"
    [PTB-XL] `yaml_fname`: "./examples/configs/example_ptbxl_vibcreg.yaml"
    """
    stream = open(yaml_fname, 'r')
    cf = yaml.load(stream, Loader=yaml.FullLoader)  # config
    return cf


def build_data_pipeline(config_dataset) -> (DataLoader, DataLoader, DataLoader):
    """
    :param config_dataset: dataset hyper-parameter settings loaded by `yaml`.
    :return: `train_data_loader`, `val_data_loader`, `test_data_loader`
    """
    cf = config_dataset
    dataset_name = cf["dataset_name"]
    batch_size = cf["batch_size"]
    num_workers = cf["num_workers"]

    if dataset_name == "UCR":
        dataset_importer = DatasetImporter(**cf)
        augs = Augmentations(**cf)
        train_dataset = UCRDataset("train", dataset_importer, augs, **cf)
        test_dataset = UCRDataset("test", dataset_importer, augs, **cf)
        # build the `DataLoader`s
        train_data_loader = DataLoader(train_dataset, batch_size, num_workers=num_workers, shuffle=True)
        val_data_loader = DataLoader(test_dataset, batch_size, num_workers=num_workers, shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size, num_workers=num_workers, shuffle=True)

    elif dataset_name == "PTB-XL":
        augs = Augmentations(**cf)
        train_dataset = PTB_XL("train", augs, **cf)
        val_dataset = PTB_XL("validate", augs, **cf)
        test_dataset = PTB_XL("test", augs, **cf)
        # build the `DataLoader`s
        train_data_loader = DataLoader(train_dataset, batch_size, num_workers=num_workers, shuffle=True)
        val_data_loader = DataLoader(val_dataset, batch_size, num_workers=num_workers, shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size, num_workers=num_workers, shuffle=True)

    else:
        raise ValueError("invalid `dataset_name`.")

    return train_data_loader, val_data_loader, test_data_loader
