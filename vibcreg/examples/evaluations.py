"""
Run by using
`python -m vibcreg.examples.evaluations`
"""

from argparse import ArgumentParser
from torch.optim import AdamW

from vibcreg.wrapper.data_pipeline_wrapper import load_hyper_param_settings, build_data_pipeline
from vibcreg.wrapper.evaluator_building_wrapper import EvaluatorBuilder


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config_dataset', type=str, help="Path to the dataset config.", default="vibcreg/configs/datasets/config_ucr.yaml")
    parser.add_argument('--config_framework', type=str, help="Path to the framework config.", default="vibcreg/configs/frameworks/config_vibcreg.yaml")
    parser.add_argument('--config_eval', type=str, help="Path to the evaluation config.", default="vibcreg/configs/evaluation/config_eval.yaml")

    parser.add_argument('--evaluation_type', default="linear_evaluation", help="linear_evaluation / fine_tuning_evaluation", type=str)
    parser.add_argument('--loading_checkpoint_fname', default="vibcreg/checkpoints/checkpoint-vibcreg-ep_100.pth")

    parser.add_argument('--device_ids', default="0", help="[GPU] a list of gpu device ids to use.", type=lambda s: [int(item.strip()) for item in s.split(',')])
    parser.add_argument('--use_wandb', default=True, help="whether to use weights and biases.", type=lambda s: eval(s))
    return parser.parse_args()


if __name__ == "__main__":
    args = load_args()

    # configs
    config_dataset = load_hyper_param_settings(args.config_dataset)
    config_framework = load_hyper_param_settings(args.config_framework)
    config_eval = load_hyper_param_settings(args.config_eval)

    # data pipeline
    train_data_loader, val_data_loader, test_data_loader = build_data_pipeline(config_dataset)

    # evaluator
    evaluator_builder = EvaluatorBuilder(config_dataset, config_framework, config_eval,
                                         train_data_loader, val_data_loader, test_data_loader,
                                         args)
    evaluator = evaluator_builder.build()

    # load encoder
    evaluator.load_encoder()

    # build classifier
    evaluator.build_classifier()

    # initialize wandb
    evaluator.init_wandb()
    evaluator.wandb_watch()

    # optimizer
    optimizer = AdamW(evaluator.trainable_params(), weight_decay=config_eval["weight_decay_ev"][args.evaluation_type])
    evaluator.setup_lr_scheduler(optimizer, "CosineAnnealingLR", train_dataset_size=train_data_loader.dataset.__len__())

    # fit
    evaluator.fit(optimizer)
