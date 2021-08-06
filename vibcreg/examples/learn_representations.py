"""
Run by using
`python -m vibcreg.examples.learn_representations --framework_config framework_filepath --dataset_config dataset_filepath`
"""

from torch.optim import AdamW
from vibcreg.backbone.resnet import ResNet1D
from vibcreg.wrapper.data_pipeline_wrapper import load_hyper_param_settings, build_data_pipeline
from vibcreg.wrapper.model_building_wrapper import build_model
from vibcreg.wrapper.run_wrapper import run_ssl_for_rl

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--framework_config', type=str, help="Path to the framework config.")
    parser.add_argument('--dataset_config', type=str, help="Path to the dataset config.")
    args = parser.parse_args()

    framework_config = load_hyper_param_settings(args.framework_config)
    dataset_config = ...


    # data pipeline
    train_data_loader, val_data_loader, test_data_loader = build_data_pipeline(dataset_config)

    # build model (encoder + SSL framework)
    encoder = ResNet1D(**framework_config)
    rl_model, rl_util = build_model(framework_config, encoder)

    # optimizer
    optimizer = AdamW(rl_model.parameters(), lr=framework_config["lr"], weight_decay=framework_config["weight_decay"])
    rl_util.setup_lr_scheduler(optimizer, kind="CosineAnnealingLR", train_dataset_size=train_data_loader.dataset.__len__())

    # W&B
    rl_util.init_wandb(framework_config, dataset_config)
    rl_util.wandb_watch()

    # run SSL for RL
    run_ssl_for_rl(framework_config, train_data_loader, val_data_loader, test_data_loader, rl_util, optimizer)
