
import tarfile

import gdown

from vibcreg.util import get_git_root


def download_ptb_xl_dataset() -> None:
    """
    check if the PTB-XL dataset exists. If not, it downloads the dataset by using the `gdown` library.
    """
    # check
    git_root = get_git_root()
    data_path = git_root.joinpath("vibcreg", "data", "PTB-XL")
    isthere = data_path.exists()

    if isthere:
        print("PTB-XL dataset already exists")
        return None

    # download
    url = "https://drive.google.com/u/0/uc?id=1qwZseccsjvvOdE17hiDL_gDmzYpaNRY0&export=download"
    output = data_path.parent.joinpath("PTB-XL.tar")
    if not output.exists():
        with output.open("wb") as ww:
            gdown.download(url, ww)

    # extract
    with tarfile.open(output) as ff:
        ff.extractall(data_path.parent)

    output.unlink()


def download_ucr_datasets() -> None:
    """
    check if the UCR datasets exist. If not, it downloads the UCR datasets by using the `gdown` library.
    """
    # check
    git_root = get_git_root()
    data_path = git_root.joinpath("vibcreg", "data", "UCRArchive_2018")
    isthere = data_path.exists()

    if isthere:
        print("UCR dataset already exists")
        return None

    # download
    url = "https://drive.google.com/u/0/uc?id=1ZvKoPvqfvZUmT05g_7ZN9uHrat-puIMj&export=download"
    output = data_path.parent.joinpath("UCRdata.tar")
    if not output.exists():
        with output.open("wb") as ww:
            gdown.download(url, ww)

    # extract
    with tarfile.open(output) as ff:
        ff.extractall(data_path.parent)

    output.unlink()


if __name__ == "__main__":
    download_ptb_xl_dataset()
    download_ucr_datasets()
