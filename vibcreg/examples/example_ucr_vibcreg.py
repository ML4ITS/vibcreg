import os
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from vibcreg.preprocess.augmentations import Augmentations
from vibcreg.preprocess.preprocess_ucr import DatasetImporter, UCRDataset

from vibcreg.backbone.resnet import ResNet1D
from vibcreg.frameworks.vibcreg_ import VIbCReg, Utility_VIbCReg

os.chdir("../")  # move to the root dir

# load hyper-parameter settings
stream = open("./examples/configs/example_ucr_vibcreg.yaml", 'r')
cf = yaml.load(stream, Loader=yaml.FullLoader)  # config

# data pipeline
dataset_importer = DatasetImporter(**cf)
augs = Augmentations(**cf)
train_dataset = UCRDataset("train", dataset_importer, augs, **cf)
test_dataset = UCRDataset("test", dataset_importer, augs, **cf)
train_data_loader = DataLoader(train_dataset, cf["batch_size"], num_workers=cf["num_workers"], shuffle=True)
val_data_loader = DataLoader(test_dataset, cf["batch_size"], num_workers=cf["num_workers"], shuffle=True)
test_data_loader = DataLoader(test_dataset, cf["batch_size"], num_workers=cf["num_workers"], shuffle=True)

# framework
encoder = ResNet1D(**cf)  # backbone-encoder
rl_model = nn.DataParallel(VIbCReg(encoder, encoder.last_channels_enc, **cf), device_ids=cf["device_ids"])
rl_util = Utility_VIbCReg(rl_model=rl_model, **cf)

# optimizer
optimizer = AdamW(rl_model.parameters(), lr=cf["lr"], weight_decay=cf["weight_decay"])
rl_util.setup_lr_scheduler(optimizer, kind="CosineAnnealingLR", train_dataset_size=train_dataset.__len__())

# W&B
rl_util.init_wandb(config=cf)
rl_util.wandb_watch()

# run SSL for RL
for epoch in range(1, cf["n_epochs"] + 1):
    rl_util.update_epoch(epoch)
    train_loss = rl_util.representation_learning(train_data_loader, optimizer, "train")
    val_loss = rl_util.validate(val_data_loader, optimizer, **cf)
    rl_util.print_train_status(epoch, optimizer.param_groups[0]['lr'], train_loss, val_loss)
    rl_util.status_log(epoch, optimizer.param_groups[0]['lr'], train_loss, val_loss)
    rl_util.save_checkpoint(epoch, optimizer, train_loss, val_loss)

    # log: feature histogram, tsne-analysis, etc. on `test_data_loader`
    if epoch in cf["tsne_analysis_log_epochs"]:
        rl_util.get_batch_of_representations(test_data_loader, **cf)  # stored internally
        rl_util.log_feature_histogram()
        rl_util.log_tsne_analysis()
        # rl_util.log_cross_correlation_matrix(train_data_loader) if cf["framework_type"] == "barlow_twins" else None

test_loss = rl_util.test(test_data_loader, optimizer)
rl_util.test_log(test_loss)
rl_util.finish_wandb()
