import torch.nn as nn

from vibcreg.backbone.resnet import ResNet1D
from vibcreg.frameworks.vibcreg_ import VIbCReg, Utility_VIbCReg


def build_model(cf):
    """
    :param cf: consists of hyper-parameter settings loaded by `yaml`.
    :return: `rl_model`, `rl_util`

    build a model (= encoder + SSL framework).
    """
    framework_type = cf["framework_type"]

    # backbone-encoder
    encoder = ResNet1D(**cf)

    # framework
    if framework_type == "vibcreg":
        rl_model = nn.DataParallel(VIbCReg(encoder, encoder.last_channels_enc, **cf), device_ids=cf["device_ids"])
        rl_util = Utility_VIbCReg(rl_model=rl_model, **cf)
    elif framework_type == "vbibcreg":
        pass
    else:
        raise ValueError("invalid `framework_type`.")

    return rl_model, rl_util
