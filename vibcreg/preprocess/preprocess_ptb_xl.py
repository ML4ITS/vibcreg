import os
import tarfile


def download_ptb_xl_dataset():
    """
    check if the PTB-XL dataset exists. If not, it downloads the dataset by using the `gdown` library.
    """
    # check
    isthere = os.path.isdir("./data/UCRArchive_2018")

    if isthere:
        return None

    # download
    url = "https://drive.google.com/u/0/uc?id=1ZvKoPvqfvZUmT05g_7ZN9uHrat-puIMj&export=download"
    output = "./data/UCRdata.tar"
    gdown.download(url, output)

    # extract
    my_tar = tarfile.open("./data/UCRdata.tar")
    my_tar.extractall("./data/")
    my_tar.close()

    os.unlink("./data/UCRdata.tar")