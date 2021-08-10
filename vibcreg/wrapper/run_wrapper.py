

def run_ssl_for_rl(args, config_dataset, config_framework,
                   train_data_loader, val_data_loader, test_data_loader,
                   rl_util, optimizer):
    """
    run self-supervised learning (SSL) for representation learning (RL).
    """
    dataset_name = config_dataset['dataset_name']
    framework_type = config_framework["framework_type"]
    n_neighbors_kNN = config_framework['n_neighbors_kNN'].get(dataset_name, 5)
    n_jobs_for_kNN = config_framework['n_jobs_for_kNN']
    model_saving_epochs = config_framework['model_saving_epochs']

    for epoch in range(1, config_framework["n_epochs"] + 1):
        rl_util.update_epoch(epoch)
        train_loss = rl_util.representation_learning(train_data_loader, optimizer, 'train')
        val_loss = rl_util.validate(val_data_loader, optimizer, dataset_name, n_neighbors_kNN, n_jobs_for_kNN)
        rl_util.print_train_status(epoch, optimizer.param_groups[0]['lr'], train_loss, val_loss)
        rl_util.status_log(epoch, optimizer.param_groups[0]['lr'], train_loss, val_loss)
        rl_util.save_checkpoint(epoch, optimizer, train_loss, val_loss, model_saving_epochs)

        # log: feature histogram, tsne-analysis, etc. on `test_data_loader`
        if epoch in args.tsne_analysis_log_epochs:
            rl_util.get_batch_of_representations(test_data_loader, dataset_name)  # stored internally
            rl_util.log_feature_histogram()
            rl_util.log_tsne_analysis()

            if framework_type == "barlow_twins":
                rl_util.log_cross_correlation_matrix(train_data_loader)
            elif framework_type == "cpc":
                rl_util.log_accuracy_cpc()

    test_loss = rl_util.test(test_data_loader, optimizer)
    rl_util.test_log(test_loss)
    rl_util.finish_wandb()
