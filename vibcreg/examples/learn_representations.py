"""
Run by using
`python -m vibcreg.examples.learn_representations --config_dataset config_dataset_filepath --config_framework config_framework_filepath --device_ids "0"`
"""

from torch.optim import AdamW

from vibcreg.wrapper.data_pipeline_wrapper import load_hyper_param_settings, build_data_pipeline
from vibcreg.wrapper.model_building_wrapper import ModelBuilder
from vibcreg.wrapper.run_wrapper import run_ssl_for_rl


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config_dataset', type=str, help="Path to the dataset config.", default="vibcreg/configs/datasets/config_ucr.yaml")
    parser.add_argument('--config_framework', type=str, help="Path to the framework config.", default="vibcreg/configs/frameworks/config_vibcreg.yaml")
    parser.add_argument('--device_ids', default="0", help="[GPU] a list of gpu device ids to use.", type=lambda s: [int(item.strip()) for item in s.split(',')])
    parser.add_argument('--use_wandb', default=True, help="whether to use weights and biases.", type=lambda s: eval(s))
    parser.add_argument('--tsne_analysis_log_epochs', default="1, 10, 20, 30, 40, 50, 100", help="[miscellaneous]", type=lambda s: [int(item.strip()) for item in s.split(',')])
    return parser.parse_args()


if __name__ == "__main__":
    from argparse import ArgumentParser
    # os.chdir("../")

    args = load_args()

    # configs
    config_dataset = load_hyper_param_settings(args.config_dataset)
    config_framework = load_hyper_param_settings(args.config_framework)

    # data pipeline
    train_data_loader, val_data_loader, test_data_loader = build_data_pipeline(config_dataset)

    # build model (encoder + SSL framework)
    model_builder = ModelBuilder(config_dataset, config_framework, args.device_ids, args.use_wandb)
    encoder = model_builder.build_encoder()
    rl_model, rl_util = model_builder.build_model(encoder)

    # optimizer
    optimizer = AdamW(rl_model.parameters(), lr=config_framework['lr'], weight_decay=config_framework['weight_decay'])
    rl_util.setup_lr_scheduler(optimizer, config_dataset['batch_size'], config_framework['n_epochs'], kind="CosineAnnealingLR", train_dataset_size=train_data_loader.dataset.__len__())

    # W&B
    rl_util.init_wandb(config_dataset, config_framework)
    rl_util.wandb_watch()

    # run SSL for RL
    run_ssl_for_rl(args, config_dataset, config_framework,
                   train_data_loader, val_data_loader, test_data_loader,
                   rl_util, optimizer)
