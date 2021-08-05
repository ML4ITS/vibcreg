import os
from torch.optim import AdamW
from vibcreg.backbone.resnet import ResNet1D
from vibcreg.wrapper.data_pipeline_wrapper import load_hyper_param_settings, build_data_pipeline
from vibcreg.wrapper.model_building_wrapper import build_model
from vibcreg.wrapper.run_wrapper import run_ssl_for_rl
os.chdir("../")  # move to the root dir

# load hyper-parameter settings
cf = load_hyper_param_settings("./configs/config_ucr_vibcreg.yaml")  # config

# data pipeline
train_data_loader, val_data_loader, test_data_loader = build_data_pipeline(cf)

# build model (encoder + SSL framework)
encoder = ResNet1D(**cf)
rl_model, rl_util = build_model(cf, encoder)

# optimizer
optimizer = AdamW(rl_model.parameters(), lr=cf["lr"], weight_decay=cf["weight_decay"])
rl_util.setup_lr_scheduler(optimizer, kind="CosineAnnealingLR", train_dataset_size=train_data_loader.dataset.__len__())

# W&B
rl_util.init_wandb(cf)
rl_util.wandb_watch()

# run SSL for RL
run_ssl_for_rl(cf, train_data_loader, val_data_loader, test_data_loader, rl_util, optimizer)
