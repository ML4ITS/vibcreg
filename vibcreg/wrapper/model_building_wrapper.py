import torch.nn as nn

from vibcreg.backbone.apc_encoder import APCEncoder
from vibcreg.backbone.downsampling_cnn import DownsamplingCNN
from vibcreg.backbone.resnet import ResNet1D
from vibcreg.frameworks.apc import APC, Utility_APC
from vibcreg.frameworks.barlow_twins import BarlowTwins, Utility_BarlowTwins
from vibcreg.frameworks.cpc import CPC, Utility_CPC
from vibcreg.frameworks.rand_init import RandInit, Utility_RandInit
from vibcreg.frameworks.simsiam import SimSiam, Utility_SimSiam
from vibcreg.frameworks.vibcreg_ import VIbCReg, Utility_VIbCReg
from vibcreg.frameworks.simclr import SimCLR, Utility_SimCLR
from vibcreg.frameworks.byol import BYOL, Utility_BYOL
from vibcreg.frameworks.tnc import TNC, Utility_TNC
from vibcreg.frameworks.vibcreg_rcd import VIbCRegRCD, Utility_VIbCRegRCD
from vibcreg.frameworks.vnibcreg import VNIbCReg, Utility_VNIbCReg
from vibcreg.frameworks.vibcreg_simclr import VIbCRegSimCLR, Utility_VIbCRegSimCLR


class ModelBuilder(object):
    def __init__(self, config_dataset, config_framework, device_ids, use_wandb):
        self.config_dataset = config_dataset
        self.config_framework = config_framework
        self.device_ids = device_ids
        self.use_wandb = use_wandb

    def build_encoder(self):
        backbone_type = self.config_framework["backbone_type"]
        if backbone_type == "resnet1d":
            encoder = ResNet1D(self.config_dataset["n_data_channels"], **self.config_framework)
        elif backbone_type == "downsampling_cnn":
            encoder = DownsamplingCNN(**self.config_framework)
        elif backbone_type == "apc_encoder":
            encoder = APCEncoder(self.config_dataset["n_data_channels"], **self.config_framework)
        else:
            raise ValueError("invalid `backbone_type`")
        return encoder

    def build_model(self, encoder, apply_data_parallel=True):
        """
        :param encoder: (backbone) encoder
        :param apply_data_parallel: whether to apply `nn.DataParallel(..)`
        :return: `rl_model`, `rl_util`

        model = encoder + (SSL) framework
        """
        # framework_cf = framework_config
        device_ids = self.device_ids
        framework_type = self.config_framework["framework_type"]
        use_wandb = self.use_wandb

        # framework
        if framework_type == "rand_init":
            rl_model = RandInit(encoder)
            if apply_data_parallel:
                rl_model = nn.DataParallel(rl_model, device_ids=device_ids)
            rl_util = Utility_RandInit(rl_model=rl_model, device_ids=device_ids, use_wandb=use_wandb,
                                       **self.config_framework)
        elif (framework_type == "vibcreg") or (framework_type == "vbibcreg"):
            rl_model = VIbCReg(encoder, encoder.last_channels_enc, **self.config_framework)
            if apply_data_parallel:
                rl_model = nn.DataParallel(rl_model, device_ids=device_ids)
            rl_util = Utility_VIbCReg(rl_model=rl_model, device_ids=device_ids, use_wandb=use_wandb,
                                      batch_size=self.config_dataset["batch_size"], **self.config_framework)
        elif framework_type == "barlow_twins":
            rl_model = BarlowTwins(encoder, encoder.last_channels_enc, **self.config_framework)
            if apply_data_parallel:
                rl_model = nn.DataParallel(rl_model, device_ids=device_ids)
            rl_util = Utility_BarlowTwins(rl_model=rl_model, device_ids=device_ids, use_wandb=use_wandb,
                                          **self.config_framework)
        elif framework_type == "simsiam":
            rl_model = SimSiam(encoder, encoder.last_channels_enc, **self.config_framework)
            if apply_data_parallel:
                rl_model = nn.DataParallel(rl_model, device_ids=device_ids)
            rl_util = Utility_SimSiam(rl_model=rl_model, device_ids=device_ids, use_wandb=use_wandb,
                                      **self.config_framework)
        elif framework_type == "cpc":
            rl_model = CPC(encoder, **self.config_framework)
            if apply_data_parallel:
                rl_model = nn.DataParallel(rl_model, device_ids=device_ids)
            rl_util = Utility_CPC(rl_model=rl_model, device_ids=device_ids, use_wandb=use_wandb,
                                  **self.config_framework)
        elif framework_type == "apc":
            rl_model = APC(encoder, **self.config_framework)
            if apply_data_parallel:
                rl_model = nn.DataParallel(rl_model, device_ids=device_ids)
            rl_util = Utility_APC(rl_model=rl_model, device_ids=device_ids, use_wandb=use_wandb,
                                  **self.config_framework)
        elif framework_type == "tnc":
            rl_model = TNC(encoder, encoder.last_channels_enc, **self.config_framework)
            if apply_data_parallel:
                rl_model = nn.DataParallel(rl_model, device_ids=device_ids)
            rl_util = Utility_TNC(rl_model=rl_model, device_ids=device_ids, use_wandb=use_wandb,
                                  batch_size=self.config_dataset["batch_size"], **self.config_framework)
        elif framework_type == "simclr":
            rl_model = SimCLR(encoder, encoder.last_channels_enc, **self.config_framework)
            if apply_data_parallel:
                rl_model = nn.DataParallel(rl_model, device_ids=device_ids)
            rl_util = Utility_SimCLR(rl_model=rl_model, device_ids=device_ids, use_wandb=use_wandb,
                                     batch_size=self.config_dataset["batch_size"], **self.config_framework)
        elif framework_type == "byol":
            rl_model = BYOL(encoder, encoder.last_channels_enc, **self.config_framework)
            if apply_data_parallel:
                rl_model = nn.DataParallel(rl_model, device_ids=device_ids)
            rl_util = Utility_BYOL(rl_model=rl_model, device_ids=device_ids, use_wandb=use_wandb,
                                   batch_size=self.config_dataset["batch_size"], **self.config_framework)
        elif framework_type == "vibcreg_rcd":
            rl_model = VIbCRegRCD(encoder, encoder.last_channels_enc, **self.config_framework)
            if apply_data_parallel:
                rl_model = nn.DataParallel(rl_model, device_ids=device_ids)
            rl_util = Utility_VIbCRegRCD(rl_model=rl_model, device_ids=device_ids, use_wandb=use_wandb,
                                         batch_size=self.config_dataset["batch_size"], **self.config_framework)
        elif framework_type == "vnibcreg":
            rl_model = VNIbCReg(encoder, encoder.last_channels_enc, **self.config_framework)
            if apply_data_parallel:
                rl_model = nn.DataParallel(rl_model, device_ids=device_ids)
            rl_util = Utility_VNIbCReg(rl_model=rl_model, device_ids=device_ids, use_wandb=use_wandb,
                                       batch_size=self.config_dataset["batch_size"], **self.config_framework)
        elif framework_type == "vibcreg_simclr":
            rl_model = VIbCRegSimCLR(encoder, encoder.last_channels_enc, **self.config_framework)
            if apply_data_parallel:
                rl_model = nn.DataParallel(rl_model, device_ids=device_ids)
            rl_util = Utility_VIbCRegSimCLR(rl_model=rl_model, device_ids=device_ids, use_wandb=use_wandb,
                                            batch_size=self.config_dataset["batch_size"], **self.config_framework)
        else:
            raise ValueError("invalid `framework_type`")

        return rl_model, rl_util
