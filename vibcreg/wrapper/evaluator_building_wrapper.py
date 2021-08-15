from vibcreg.evaluation.evaluator_ucr import EvaluatorUCR
from vibcreg.evaluation.evaluator_ptbxl import EvaluatorPTB_XL


class EvaluatorBuilder(object):
    def __init__(self, config_dataset, config_framework, config_eval,
                 train_data_loader, val_data_loader, test_data_loader,
                 args):
        self.config_dataset = config_dataset
        self.config_framework = config_framework
        self.config_eval = config_eval
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.args = args

    def build(self):
        dataset_name = self.config_dataset["dataset_name"]

        if dataset_name == "UCR":
            evaluator = EvaluatorUCR(config_dataset=self.config_dataset, config_framework=self.config_framework, config_eval=self.config_eval,
                                     train_data_loader=self.train_data_loader, val_data_loader=self.val_data_loader, test_data_loader=self.test_data_loader,
                                     evaluation_type=self.args.evaluation_type, loading_checkpoint_fname=self.args.loading_checkpoint_fname, device_ids=self.args.device_ids, use_wandb=self.args.use_wandb)
        elif dataset_name == "PTB-XL":
            evaluator = EvaluatorPTB_XL(config_dataset=self.config_dataset, config_framework=self.config_framework, config_eval=self.config_eval,
                                        train_data_loader=self.train_data_loader, val_data_loader=self.val_data_loader, test_data_loader=self.test_data_loader,
                                        evaluation_type=self.args.evaluation_type, loading_checkpoint_fname=self.args.loading_checkpoint_fname, device_ids=self.args.device_ids, use_wandb=self.args.use_wandb)
        else:
            raise ValueError("invalid `dataset_name`")

        return evaluator
