"""
Run SSL and Evaluation on UCR
to collect all the experimental results.

`python -m vibcreg.examples.ssl_eval_ucr --config_dataset config_dataset_filepath --config_framework config_framework_filepath --device_ids "0"`
"""
from torch.optim import AdamW

from vibcreg.wrapper.data_pipeline_wrapper import load_hyper_param_settings, build_data_pipeline
from vibcreg.wrapper.model_building_wrapper import ModelBuilder
from vibcreg.wrapper.run_wrapper import run_ssl_for_rl
from vibcreg.wrapper.evaluator_building_wrapper import EvaluatorBuilder


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config_dataset', type=str, help="Path to the dataset config.",
                        default="vibcreg/configs/datasets/config_ucr.yaml")
    parser.add_argument('--config_framework', type=str, help="Path to the framework config.",
                        default="vibcreg/configs/frameworks/config_simclr.yaml")
    parser.add_argument('--device_ids', default="0", help="[GPU] a list of gpu device ids to use.", type=lambda s: [int(item.strip()) for item in s.split(',')])
    parser.add_argument('--use_wandb', default=True, help="whether to use weights and biases.", type=lambda s: eval(s))
    parser.add_argument('--tsne_analysis_log_epochs', default="1, 10, 20, 30, 40, 50, 100", help="[miscellaneous]", type=lambda s: [int(item.strip()) for item in s.split(',')])

    parser.add_argument('--evaluation_type', default="linear_evaluation",
                        help="linear_evaluation / fine_tuning_evaluation", type=str)
    parser.add_argument('--loading_checkpoint_fname',
                        default="vibcreg/checkpoints/checkpoint-simclr-ep_100.pth")

    return parser.parse_args()


if __name__ == "__main__":
    from argparse import ArgumentParser
    args = load_args()
    # configs
    config_dataset = load_hyper_param_settings(args.config_dataset)
    config_framework = load_hyper_param_settings(args.config_framework)
    config_eval = load_hyper_param_settings("vibcreg/configs/evaluation/config_eval.yaml")

    # 'DatsetName': [crop_size, seq_full_len, n_classes]
    ucr_dataset_names = {'ShapesAll': [256, 512, 60],
                         'FiftyWords': [135, 270, 50],
                         'NonInvasiveFetalECGThorax1': [750, 750, 42],
                         'Phoneme': [512, 1024, 39],
                         'WordSynonyms': [135, 270, 25],
                         'Crop': [46, 46, 24],
                         'ElectricDevices': [48, 96, 7],
                         'StarLightCurves': [512, 1024, 3],
                         'Wafer': [152, 152, 2],
                         'ECG5000': [70, 140, 5],
                         'TwoPatterns': [64, 128, 4],
                         'FordA': [250, 500, 2],
                         'UWaveGestureLibraryAll': [473, 945, 8],
                         'FordB': [250, 500, 2],
                         'ChlorineConcentration': [166, 166, 3],
                         }

    if config_dataset['is_tnc_used']:
        # adjust crop_size given for TNC-based models (it needs some extra temporal space)
        for k in ucr_dataset_names.keys():
            item = ucr_dataset_names[k]  # [crop_size, seq_full_len, n_classes]
            item[0] = item[1] // 3  # adjust `crop_size`
            ucr_dataset_names[k] = item

        # rand_seeds = [0, 1, 2, 3, 4]
    rand_seeds = [0, 1, 2]
    for rand_seed in rand_seeds:
        for ucr_dataset_name, (crop_size, seq_full_len, n_classes) in ucr_dataset_names.items():

            # change of configs
            config_dataset['ucr_dataset_name'] = ucr_dataset_name
            config_dataset['subseq_len'] = crop_size
            config_dataset['train_random_seed'] = rand_seed
            config_dataset['test_random_seed'] = rand_seed
            config_dataset['used_augmentations'] = ['RC', 'AmpR', 'Vshift']
            config_dataset['train_data_ratio'] = 0.8

            # data pipeline
            train_data_loader, val_data_loader, test_data_loader = build_data_pipeline(config_dataset)

            # build model (encoder + SSL framework)
            model_builder = ModelBuilder(config_dataset, config_framework, args.device_ids, args.use_wandb)
            encoder = model_builder.build_encoder()
            rl_model, rl_util = model_builder.build_model(encoder)

            # optimizer
            optimizer = AdamW(rl_model.parameters(), lr=config_framework['lr'], weight_decay=config_framework['weight_decay'])
            rl_util.setup_lr_scheduler(optimizer, config_dataset['batch_size'], config_framework['n_epochs'],
                                       kind="CosineAnnealingLR",
                                       train_dataset_size=train_data_loader.dataset.__len__())

            # W&B
            rl_util.init_wandb(config_dataset, config_framework)
            rl_util.wandb_watch()

            # run SSL for RL
            run_ssl_for_rl(args, config_dataset, config_framework,
                           train_data_loader, val_data_loader, test_data_loader,
                           rl_util, optimizer)

            # ==========================================================================
            # LE
            config_dataset['subseq_len'] = seq_full_len
            config_dataset['used_augmentations'] = ['AmpR', 'Vshift']
            config_dataset['train_data_ratio'] = 0.8
            args.evaluation_type = 'linear_evaluation'

            # data pipeline
            train_data_loader, val_data_loader, test_data_loader = build_data_pipeline(config_dataset)

            # evaluator
            evaluator_builder = EvaluatorBuilder(config_dataset, config_framework, config_eval,
                                                 train_data_loader, val_data_loader, test_data_loader,
                                                 args)
            evaluator = evaluator_builder.build()

            # load encoder
            evaluator.load_encoder(pretrained_encoder=rl_model.module.encoder)

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

            # ==========================================================================
            # FT
            FT_cases = {'ShapesAll': [0.1, 0.2],
                        'FiftyWords': [0.2],
                        'NonInvasiveFetalECGThorax1': [0.05, 0.1, 0.2],
                        'Phoneme': [0.2],
                        'WordSynonyms': [0.1, 0.2],
                        }

            if ucr_dataset_name in FT_cases.keys():
                ratios = FT_cases[ucr_dataset_name]
            else:
                ratios = [0.05, 0.1, 0.2]  # 5%, 10%, 20%

            for ratio in ratios:
                config_dataset['subseq_len'] = seq_full_len
                config_dataset['used_augmentations'] = ['AmpR', 'Vshift']
                config_dataset['train_data_ratio'] = 0.8 * ratio
                args.evaluation_type = 'fine_tuning_evaluation'

                # data pipeline
                train_data_loader, val_data_loader, test_data_loader = build_data_pipeline(config_dataset)

                # evaluator
                evaluator_builder = EvaluatorBuilder(config_dataset, config_framework, config_eval,
                                                     train_data_loader, val_data_loader, test_data_loader,
                                                     args)
                evaluator = evaluator_builder.build()

                # load encoder
                evaluator.load_encoder(pretrained_encoder=rl_model.module.encoder)

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

