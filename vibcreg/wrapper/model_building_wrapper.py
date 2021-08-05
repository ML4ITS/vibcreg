import torch.nn as nn

from vibcreg.frameworks.vibcreg_ import VIbCReg, Utility_VIbCReg
from vibcreg.frameworks.barlow_twins import BarlowTwins, Utility_BarlowTwins
from vibcreg.frameworks.simsiam import SimSiam, Utility_SimSiam
from vibcreg.frameworks.rand_init import RandInit, Utility_RandInit


def build_model(cf, encoder):
    """
    :param cf: consists of hyper-parameter settings loaded by `yaml`.
    :param encoder: (backbone) encoder
    :return: `rl_model`, `rl_util`

    build a model (= encoder + SSL framework).
    """
    framework_type = cf["framework_type"]

    # framework
    if framework_type == "rand_init":
        rl_model = nn.DataParallel(RandInit(encoder), device_ids=cf["device_ids"])
        rl_util = Utility_RandInit(rl_model=rl_model, **cf)
    elif (framework_type == "vibcreg") or (framework_type == "vbibcreg"):
        rl_model = nn.DataParallel(VIbCReg(encoder, encoder.last_channels_enc, **cf), device_ids=cf["device_ids"])
        rl_util = Utility_VIbCReg(rl_model=rl_model, **cf)
    elif framework_type == "barlow_twins":
        rl_model = nn.DataParallel(BarlowTwins(encoder, encoder.last_channels_enc, **cf), device_ids=cf["device_ids"])
        rl_util = Utility_BarlowTwins(rl_model=rl_model, **cf)
    elif framework_type == "simsiam":
        rl_model = nn.DataParallel(SimSiam(encoder, encoder.last_channels_enc, **cf), device_ids=cf["device_ids"])
        rl_util = Utility_SimSiam(rl_model=rl_model, **cf)
    else:
        raise ValueError("invalid `framework_type`")

    return rl_model, rl_util
