
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
        return None

    # download
    url = "https://drive.google.com/u/0/uc?id=1qwZseccsjvvOdE17hiDL_gDmzYpaNRY0&export=download"
    output = data_path.parent.joinpath("PTB-XL.tar")
    with output.open("wb") as ww:
        gdown.download(url, ww)

    # extract
    with tarfile.open(output) as ff:
        ff.extractall(data_path.parent)

    output.unlink()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset")
    args = parser.parse_args()

    if args.dataset == "ptbxl":
        download_ptb_xl_dataset()
