"""
You can run either 'linear evaluation' or 'fine tuning evaluation'.
"""
import os

from torch.optim import AdamW

from vibcreg.backbone.resnet import ResNet1D
from vibcreg.evaluation.evaluator_ptbxl import EvaluatorPTB_XL
from vibcreg.evaluation.evaluator_skeleton import update_config
from vibcreg.wrapper.data_pipeline_wrapper import load_hyper_param_settings, build_data_pipeline

os.chdir("../../")  # move to the root dir

# load hyper-parameter settings
evaluation_type = "fine_tuning_evaluation"  # "linear_evaluation" / "fine_tuning_evaluation"
cf = load_hyper_param_settings("./configs/config_ptbxl_vibcreg.yaml")  # config
update_config(cf, evaluation_type=evaluation_type)

# data pipeline
train_data_loader, val_data_loader, test_data_loader = build_data_pipeline(cf)

# evaluator
evaluator = EvaluatorPTB_XL(cf=cf,
                            train_data_loader=train_data_loader,
                            val_data_loader=val_data_loader,
                            test_data_loader=test_data_loader, **cf)

# load model
encoder = ResNet1D(**cf)
evaluator.load_model(encoder)

# build classifier
evaluator.build_classifier(**cf)

# initialize wandb
evaluator.init_wandb(**cf)
evaluator.wandb_watch()

# optimizer
optimizer = AdamW(evaluator.trainable_params(**cf), weight_decay=cf["weight_decay_ev"][cf["evaluation_type"]])
evaluator.setup_lr_scheduler(optimizer, kind="CosineAnnealingLR", train_dataset_size=train_data_loader.dataset.__len__(), **cf)

# fit
evaluator.fit(optimizer, **cf)
