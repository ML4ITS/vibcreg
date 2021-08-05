

def run_ssl_for_rl(cf, train_data_loader, val_data_loader, test_data_loader, rl_util, optimizer):
    """
    run self-supervised learning (SSL) for representation learning (RL).
    """
    framework_type = cf["framework_type"]

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

            if framework_type == "barlow_twins":
                rl_util.log_cross_correlation_matrix(train_data_loader)
            elif framework_type == "cpc":
                rl_util.log_accuracy_cpc()

    test_loss = rl_util.test(test_data_loader, optimizer)
    rl_util.test_log(test_loss)
    rl_util.finish_wandb()
