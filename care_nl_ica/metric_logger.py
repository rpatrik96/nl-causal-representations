from dataclasses import dataclass

import torch
import torchmetrics

class Metrics(object):
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
        self.stat_scores = torchmetrics.StatScores() #FP, FN, TP, TN

    def update(self, y_pred:torch.Tensor, y_true:torch.Tensor)->None:
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

    def compute(self)->dict:
        """
        Compute the metrics.
        """
        [tp, fp, tn, fn, sup] = self.stat_scores.compute()
        panel_name = "Metrics"
        return {
            f'{panel_name}/accuracy': self.accuracy.compute(),
            # f'{panel_name}/roc_auc': self.roc_auc.compute(),
            # f'{panel_name}/auc_score': self.auc_score.compute(),
            f'{panel_name}/precision': self.precision.compute(),
            f'{panel_name}/recall': self.recall.compute(),
            f'{panel_name}/f1': self.f1.compute(),
            f'{panel_name}/hamming_distance': self.hamming_distance.compute(),
            f'{panel_name}/tp': tp,
            f'{panel_name}/fp': fp,
            f'{panel_name}/tn': tn,
            f'{panel_name}/fn': fn,
            f'{panel_name}/support': sup
        }

@dataclass
class JacobianMetrics:
    norm_diff:float
    thresholded_norm_diff:float
    optimal_threshold:float
    sparsity_accuracy:float
    amari_distance:float

def amari_distance(W:torch.Tensor, A:torch.Tensor)->float:
    """
    Computes the Amari distance between the products of two collections of matrices W and A.
    It cancels when the average of the absolute value of WA is a permutation and scale matrix.

    Based on the implementation of Amari distance in:
    https://github.com/pierreablin/picard/blob/master/picard/_tools.py

    Parameters
    ----------
    W : torch.Tensor, shape (n_features, n_features)
        Input collection of matrices
    A : torch.Tensor, shape (n_features, n_features)
        Input collection of matrices
    Returns
    -------
    d : torch.Tensor, shape (1, )
        The Amari distances between the average of absolute values of the products of W and A.
    """

    P = W@A

    def s(r):
        return ((r ** 2).sum(axis=1) / (r ** 2).max(axis=1)[0] - 1).sum()

    return ((s(P.abs()) + s(P.T.abs())) / (2 * P.shape[1])).item()


def cima_kl_diagonality(matrix:torch.Tensor)->float:
    """
    Calculates the IMA constrast (the lefy KL measure of diagonality).

    :param matrix: matrix as a torch.Tensor
    :return:
    """
    # return (torch.diag(matrix).norm('fro') / matrix.norm('fro')).item()

    return 0.5 * (torch.linalg.slogdet(torch.diag(torch.diag(matrix)))[1] -
                  torch.linalg.slogdet(matrix)[1]).item()



def ksi_correlation(hz:torch.Tensor, z:torch.Tensor)->list:
    """
    Calculates the correlation between the latent variables and the observed variables.
    :param hz: latent variables
    :param z: observed variables
    """
    num_samples = z.shape[0]
    
    # from http://arxiv.org/abs/1909.10140
    # 1. take the (zi, hzi) pairs (for each dimension),
    # sort zi and
    # use the indices that sort zi to sort hzi in ascending order
    sorted_representations = [hzi[torch.sort(zi, axis=-1)[1]] for (zi, hzi) in zip(z.T, hz.T)]
    # 2. rank the sorted sorted_representations dimensionwise (i.e.,s_repr),
    # where the rank of each item is the number of hzi_sorted s.t.
    # it counts the smaller elements that item
    representation_ranks = [torch.tensor([(s_repr <= item).sum() for item in s_repr]) for s_repr in
                            sorted_representations]
    # 3. use eq. 11  (assumes no ties - ties can be ignored for large num_samples)
    ksi = [1 - 3 * (r[1:] - r[:-1]).abs().sum() / (num_samples ** 2 - 1) for r in representation_ranks]
    
    # +1: normalize by the possible min and max values
    ksi_max = (num_samples - 2) / (num_samples + 1)
    ksi_min = -.5 + 1 / num_samples
    
    return ksi



