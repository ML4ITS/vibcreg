"""
Run SSL and Evaluation on UEA
to collect all the experimental results for all UEA datasets,
using `DatasetImporterDefault`

`python -m vibcreg.examples.ssl_eval_uea_all --config_dataset config_dataset_filepath --config_framework config_framework_filepath --device_ids "0"`
"""
import os

import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
import wandb
from torch.optim import AdamW

from vibcreg.wrapper.data_pipeline_wrapper import load_hyper_param_settings, build_data_pipeline
from vibcreg.wrapper.model_building_wrapper import ModelBuilder
from vibcreg.wrapper.run_wrapper import run_ssl_for_rl
from vibcreg.wrapper.evaluator_building_wrapper import EvaluatorBuilder

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_from_tsfile
from sklearn.preprocessing import LabelEncoder

from vibcreg.util import get_git_root


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config_dataset', type=str, help="Path to the dataset config.",
                        default="vibcreg/configs/datasets/config_uea.yaml")
    parser.add_argument('--config_framework', type=str, help="Path to the framework config.",
                        default="vibcreg/configs/frameworks/config_vibcreg.yaml")
    parser.add_argument('--device_ids', default="0", help="[GPU] a list of gpu device ids to use.", type=lambda s: [int(item.strip()) for item in s.split(',')])
    parser.add_argument('--use_wandb', default=True, help="whether to use weights and biases.", type=lambda s: eval(s))
    parser.add_argument('--tsne_analysis_log_epochs',
                        default="50,100,200,300,400,500", help="[miscellaneous]", type=lambda s: [int(item.strip()) for item in s.split(',')])

    parser.add_argument('--evaluation_type', default="linear_evaluation",
                        help="linear_evaluation / fine_tuning_evaluation", type=str)
    parser.add_argument('--loading_checkpoint_fname',
                        default="vibcreg/checkpoints/checkpoint-vibcreg-ep_100.pth")
    parser.add_argument('--n_jobs', type=int, help='n_jobs for GridSearch and kNN-clf',
                        default=8)
    return parser.parse_args()


if __name__ == "__main__":
    from argparse import ArgumentParser
    args = load_args()
    # configs
    config_dataset = load_hyper_param_settings(args.config_dataset)
    config_framework = load_hyper_param_settings(args.config_framework)
    config_eval = load_hyper_param_settings("vibcreg/configs/evaluation/config_eval.yaml")

    uea_dataset_names = os.listdir(get_git_root().joinpath("vibcreg", "data", "UEAArchive_2018"))
    uea_dataset_names = sorted(uea_dataset_names)
    print('uea_dataset_names:', uea_dataset_names, '\n\n')

    used_augmentations_origin = config_dataset['used_augmentations']

    # uea_dataset_names = uea_dataset_names[uea_dataset_names.index('FingerMovements'):]
    uea_dataset_names = ['StandWalkJump']  # NonInvasiveFetalECGThorax1
    rand_seeds = [0]
    for rand_seed in rand_seeds:
        for uea_dataset_name in uea_dataset_names:

            # temporarily omit these datasets for faster eval
            if uea_dataset_name in ['ElectricDevices']:
                continue

            # try:
            data_root = get_git_root().joinpath("vibcreg", "data", "UEAArchive_2018", uea_dataset_name)
            train_x, train_y = load_from_tsfile(str(data_root.joinpath(f'{uea_dataset_name}_TEST.ts')))
            # df_train = pd.read_csv(data_root.joinpath(f"{uea_dataset_name}_TRAIN.tsv"), sep='\t', header=None)
            # df_test = pd.read_csv(data_root.joinpath(f"{uea_dataset_name}_TEST.tsv"), sep='\t', header=None)
            seq_len = len(train_x.iloc[0, 0]) #df_test.iloc[:, 1:].shape[1]

            # set `crop_size`
            if config_dataset['is_tnc_used']:
                crop_size = seq_len // 3
            elif config_framework['framework_type'] in ['vibcreg', 'vibcreg_simclr']:
                if config_framework['length_sampling']:
                    crop_size = seq_len
            else:
                crop_size = seq_len // 2

            print('================')
            print('Dataset:', uea_dataset_name)
            #print('df_train.shape:', df_train.shape)
            print('seq_len:', seq_len)
            print('crop_size:', crop_size)
            print('================')

            # if a number of training samples are below ...
            # if len(train_x) < 200:
            #     continue

            # change of configs
            config_dataset['uea_dataset_name'] = uea_dataset_name
            config_dataset['subseq_len'] = crop_size
            config_dataset['train_random_seed'] = rand_seed
            config_dataset['test_random_seed'] = rand_seed
            config_dataset['used_augmentations'] = used_augmentations_origin
            config_dataset['n_data_channels'] = train_x.shape[1]

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
            rl_util.init_wandb(config_dataset, config_framework, overwritten_project_name='UEA-ALL-kNN')
            rl_util.wandb_watch()

            # run SSL for RL
            run_ssl_for_rl(args, config_dataset, config_framework,
                           train_data_loader, val_data_loader, test_data_loader,
                           rl_util, optimizer, include_finish_wandb=False)

            # get test acc
            config_dataset['subseq_len'] = seq_len
            config_dataset['used_augmentations'] = []
            device = torch.device(args.device_ids[0])

            # collect representations from a training set
            train_data_loader, val_data_loader, test_data_loader = build_data_pipeline(config_dataset)
            ys = None
            labels = None
            for batch in train_data_loader:
                x, _, label = batch
                with torch.no_grad():
                    y = rl_model.module.encoder(x.to(device))  # (B, D)
                y = y.cpu().numpy()
                label = label.numpy()
                if ys is None:
                    ys = y
                    labels = label
                else:
                    ys = np.concatenate((ys, y), axis=0)
                    labels = np.concatenate((labels, label), axis=0)

            # fit svm
            parameters = {'C': [10**i for i in range(-4, 5)] + [np.inf]}
            # parameters = {'C': [2 ** i for i in range(-5, 17, 2)] + [10**10],
            #               'gamma': [2 ** i for i in range(-15, 5, 2)] + ['scale'],
            #               }
            svc = svm.SVC(kernel='rbf', tol=1e-2)
            print('SVM grid search starts.')
            search = GridSearchCV(svc, parameters, n_jobs=args.n_jobs, verbose=1)
            # scaler = StandardScaler()
            # ys_scaled = scaler.fit_transform(ys)
            search.fit(ys, labels.ravel())
            print('SVM grid search ends.')
            svm_clf = search.best_estimator_
            print('best SVM:', svm_clf)

            # fit knn
            print('Fit knn classifier.')
            knn1_clf = KNeighborsClassifier(n_neighbors=1, n_jobs=args.n_jobs)
            knn3_clf = KNeighborsClassifier(n_neighbors=3, n_jobs=args.n_jobs)
            knn5_clf = KNeighborsClassifier(n_neighbors=5, n_jobs=args.n_jobs)
            knn1_clf.fit(ys, labels.ravel())
            knn3_clf.fit(ys, labels.ravel())
            knn5_clf.fit(ys, labels.ravel())
            print('End fitting knn classifier.')

            # test
            ys = None
            labels = None
            for batch in test_data_loader:
                x, _, label = batch
                with torch.no_grad():
                    y = rl_model.module.encoder(x.to(device))  # (B, D)
                y = y.cpu().numpy()
                label = label.numpy()
                if ys is None:
                    ys = y
                    labels = label
                else:
                    ys = np.concatenate((ys, y), axis=0)
                    labels = np.concatenate((labels, label), axis=0)

            # predict
            pred_labels_svm = svm_clf.predict(ys)  # pred_labels_svm = svm_clf.predict(scaler.transform(ys))
            pred_labels_knn1 = knn1_clf.predict(ys)
            pred_labels_knn3 = knn3_clf.predict(ys)
            pred_labels_knn5 = knn5_clf.predict(ys)

            # acc.
            acc_svm = accuracy_score(labels, pred_labels_svm)
            acc_knn1 = accuracy_score(labels, pred_labels_knn1)
            acc_knn3 = accuracy_score(labels, pred_labels_knn3)
            acc_knn5 = accuracy_score(labels, pred_labels_knn5)
            wandb.log({f'test_acc-svm': acc_svm,
                       'test_acc-knn_1': acc_knn1,
                       'test_acc-knn_3': acc_knn3,
                       'test_acc-knn_5': acc_knn5,
                       })

            rl_util.finish_wandb()
            # except:
            #     wandb.finish()
