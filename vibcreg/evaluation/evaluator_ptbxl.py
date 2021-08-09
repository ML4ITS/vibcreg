import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from vibcreg.evaluation.evaluator_skeleton import Evaluator


class EvaluatorPTB_XL(Evaluator):
    def __init__(self, **kwargs):
        super(EvaluatorPTB_XL, self).__init__(**kwargs)

    def _set_criterion(self):
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def _clf_in_size(self, framework_type, **kwargs) -> int:
        if framework_type == 'cpc':
            in_size = self.rl_model.module.ar.ar.input_size
        elif framework_type == "apc":
            better_context_kind_apc = kwargs.get("better_context_kind_apc", None)
            mul = len(better_context_kind_apc.split("+"))
            in_size = self.encoder.module.rnn.hidden_size * mul
        else:  # if `encoder` is `ResNet1D`
            in_size = self.encoder.module.res_blocks[-1].conv_1x1.out_channels
        return in_size

    def _clf_out_size(self):
        return 71

    def _adjust_label(self, label):
        return label

    def _out_from_encoder_classifier(self, x, **kwargs):
        """
        returns output from an encoder-classifier.
        """
        y = self.encoder(x)
        if kwargs.get("framework_type") == "apc":
            better_context_kind_apc = kwargs.get("better_context_kind_apc", None)
            y = self.encoder.module.compute_better_context(y, kind=better_context_kind_apc)

        out = self.classifier(y)  # (batch * 1)
        return out

    def _train_eval_setting_during_fit(self, status):
        """
        set the .train() / .eval() settings during fit(..) for `self.encoder` and `self.classifier`.
        """
        if status == "train":
            # encoder
            if (self.framework_type == 'supervised') or (self.framework_type == "apc"):
                self.encoder.train()
            elif self.evaluation_type == "fine_tuning_evaluation":
                self._enable_bn_stat_update_only(self.encoder)
            else:
                self.encoder.eval()
            # classifier
            self.classifier.train()

        elif (status == "validate") or (status == "test"):
            self.encoder.eval()
            if self.framework_type == "apc":
                self.encoder.train()
            self.classifier.eval()

    @torch.no_grad()
    def _compute_test_macroAUC(self, subseq_len, **kwargs):
        """
        References:
        [1] Scikit-learn, "Plot ROC curves for the multilabel problem" (https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py)
        """
        device = self.device_ids[0]

        # Collect the entire test data
        labels = torch.tensor([])  # (entire_batch * 71)
        cls_preds = torch.tensor([])  # (entire_batch * 71)
        for subx_view1, subx_view2, label in self.test_data_loader:  # subx: (batch * n_channels * subseq_len)
            labels = torch.cat((labels, label))

            if self.framework_type == 'cpc':
                z = self.encoder(subx_view1.float().to(device))  # (batch * n_channels * reduced_seq_len)
                out = z.mean(dim=2)  # (batch * n_channels)
                out = self.classifier(out).cpu().detach()  # (batch * 71)
                minibatch_cls_pred = torch.sigmoid(out)
            else:
                # averaged classification prediction
                i = 0
                minibatch_cls_preds = torch.tensor([])  # (n_slides, batch * 71)
                while subx_view1[:, :, i:i + subseq_len].shape[-1] == subseq_len:
                    subx = subx_view1[:, :, i:i + subseq_len].float()  # (batch * 12 * subseq_len)
                    z = self.encoder(subx.to(device))  # (batch * feature_size)
                    out = self.classifier(z).cpu().detach()  # (batch * 71)
                    out = torch.sigmoid(out)  # (batch * 71)
                    out = out.unsqueeze(dim=0)  # (1 * batch * 71)
                    minibatch_cls_preds = torch.cat((minibatch_cls_preds, out))  # (n_slides * batch * 71)
                    i += subseq_len
                minibatch_cls_pred = minibatch_cls_preds.mean(dim=0)  # (batch * 71)
            cls_preds = torch.cat((cls_preds, minibatch_cls_pred))  # (entire_batch * 71)

        # Compute Macro-AUC
        # - compute ROC curve and ROC area for each class
        fpr = dict()  # false positive rate
        tpr = dict()  # true positive rate
        roc_auc = dict()
        n_classes = labels.shape[-1]
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(labels[:, i], cls_preds[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # - first aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # - then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # - finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # confusion matrix
        self._log_confusion_matrix(labels, cls_preds, **kwargs)
        return roc_auc["macro"]

    def _log_confusion_matrix(self, y_test, y_pred, tsne_analysis_log_epochs, **kwargs):
        """
        [1] Stackoverflow, https://stackoverflow.com/questions/62722416/plot-confusion-matrix-for-multilabel-classifcation-python
        """
        if not self.use_wandb:
            return None
        if self.epoch not in tsne_analysis_log_epochs:
            return None

        y_pred = np.round(y_pred)  # i.e., threshold = 0.5
        n_classes = y_test.shape[-1]

        y_test = np.abs(1 - y_test)  # to make 0 False and 1 True.
        y_pred = np.abs(1 - y_pred)

        f, axes = plt.subplots(round(n_classes / 8), 8, figsize=(25, 15))
        axes = axes.ravel()
        for i in range(n_classes):
            disp = ConfusionMatrixDisplay(confusion_matrix(y_test[:, i], y_pred[:, i]), display_labels=[0, i])
            disp.plot(ax=axes[i], values_format='.4g')
            disp.ax_.set_title(f'class {i}')
            if i < 10:
                disp.ax_.set_xlabel('')
            if i % 5 != 0:
                disp.ax_.set_ylabel('')
            disp.im_.colorbar.remove()
            disp.ax_.set_xticks([])
            disp.ax_.set_yticks([])

        plt.subplots_adjust(wspace=0.10, hspace=0.5)
        f.colorbar(disp.im_, ax=axes)
        plt.suptitle(f"epoch: {self.epoch}")
        wandb.log({f'confusion_mat-ep_{self.epoch}': plt})
        plt.close()
