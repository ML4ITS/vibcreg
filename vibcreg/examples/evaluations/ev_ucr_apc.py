"""
You can run either 'linear evaluation' or 'fine tuning evaluation'.
"""
import os
from torch.optim import AdamW
from vibcreg.wrapper.data_pipeline_wrapper import load_hyper_param_settings, build_data_pipeline
from vibcreg.evaluation.evaluator import update_config, Evaluator
from vibcreg.backbone.apc_encoder import APCEncoder
os.chdir("../../")  # move to the root dir

# load hyper-parameter settings
evaluation_type = "fine_tuning_evaluation"  # "linear_evaluation" / "fine_tuning_evaluation"
cf = load_hyper_param_settings("./configs/config_ucr_apc.yaml")  # config
update_config(cf, evaluation_type=evaluation_type)

# data pipeline
train_data_loader, val_data_loader, test_data_loader = build_data_pipeline(cf)

# evaluator
evaluator = Evaluator(cf, train_data_loader, val_data_loader, test_data_loader)

# load model
encoder = APCEncoder(**cf)
evaluator.load_model(encoder, **cf)

# build classifier
evaluator.build_classifier(**cf)

# initialize wandb
evaluator.init_wandb(**cf)
evaluator.wandb_watch(**cf)

# optimizer
optimizer = AdamW(evaluator.trainable_params(**cf), weight_decay=cf["weight_decay_ev"][cf["evaluation_type"]])
evaluator.setup_lr_scheduler(optimizer, kind="CosineAnnealingLR", train_dataset_size=train_data_loader.dataset.__len__(), **cf)

# fit
evaluator.fit(optimizer, **cf)