import torch
import torchmetrics


class MetricLogger(object):
    def __init__(self):
        """
        Initialize the metrics.
        """
        super().__init__()

        self.accuracy = torchmetrics.Accuracy()
        self.roc_auc = torchmetrics.AUROC(num_classes=2)
        self.auc_score = torchmetrics.AUC()
        self.precision = torchmetrics.Precision()
        self.recall = torchmetrics.Recall()
        self.f1 = torchmetrics.F1()
        self.hamming_distance = torchmetrics.HammingDistance()
        self.stat_scores = torchmetrics.StatScores()  # FP, FN, TP, TN

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Update the metrics given the predictions and the ground truth.
        """
        self.accuracy.update(y_pred, y_true)
        self.roc_auc.update(y_pred, y_true)
        self.auc_score.update(y_pred, y_true)
        self.precision.update(y_pred, y_true)
        self.recall.update(y_pred, y_true)
        self.f1.update(y_pred, y_true)
        self.hamming_distance.update(y_pred, y_true)
        self.stat_scores.update(y_pred, y_true)

    def compute(self) -> dict:
        """
        Compute the metrics.
        """
        [tp, fp, tn, fn, sup] = self.stat_scores.compute()
        panel_name = "Metrics"
        return {
            f"{panel_name}/accuracy": self.accuracy.compute(),
            # f'{panel_name}/roc_auc': self.roc_auc.compute(),
            # f'{panel_name}/auc_score': self.auc_score.compute(),
            f"{panel_name}/precision": self.precision.compute(),
            f"{panel_name}/recall": self.recall.compute(),
            f"{panel_name}/f1": self.f1.compute(),
            f"{panel_name}/hamming_distance": self.hamming_distance.compute(),
            f"{panel_name}/tp": tp,
            f"{panel_name}/fp": fp,
            f"{panel_name}/tn": tn,
            f"{panel_name}/fn": fn,
            f"{panel_name}/support": sup,
        }
