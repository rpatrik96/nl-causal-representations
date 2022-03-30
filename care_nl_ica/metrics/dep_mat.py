from dataclasses import dataclass
from typing import Dict

import torch
from torchmetrics import Metric

from care_nl_ica.losses.dep_mat import permutation_loss
from care_nl_ica.metrics.ica_dis import amari_distance


@dataclass
class JacobianMetrics:
    norm_diff: float
    thresholded_norm_diff: float
    optimal_threshold: float
    sparsity_accuracy: float
    amari_distance: float
    permutation_quality: float

    def log_dict(self, panel_name) -> Dict[str, float]:
        return {
            f"{panel_name}/jacobian/norm_diff": self.norm_diff,
            f"{panel_name}/jacobian/thresholded_norm_diff": self.thresholded_norm_diff,
            f"{panel_name}/jacobian/optimal_threshold": self.optimal_threshold,
            f"{panel_name}/jacobian/sparsity_accuracy": self.sparsity_accuracy,
            f"{panel_name}/jacobian/permutation_quality": self.permutation_quality,
            f"{panel_name}/jacobian/amari_distance": self.amari_distance,
        }


def calc_jacobian_metrics(
    dep_mat: torch.Tensor,
    gt_jacobian_unmixing,
    indirect_causes,
    gt_jacobian_mixing_permuted,
    threshold: float = 1e-3,
) -> JacobianMetrics:

    if dep_mat.min() < 0:
        warn(
            "The prediction has negative values, taking the absolute value...",
            RuntimeWarning,
        )
        dep_mat = dep_mat.abs()

    if (dep_max := dep_mat.max()) != 1.0:
        warn("The prediction values not in [0;1], normalizing...", RuntimeWarning)
        dep_mat /= dep_max

    # calculate the optimal threshold for 1 accuracy
    # calculate the indices where the GT is 0 (in the lower triangular part)
    sparsity_mask = (torch.tril(gt_jacobian_unmixing.abs() < 1e-6)).bool()

    if sparsity_mask.sum() > 0:
        optimal_threshold = dep_mat[sparsity_mask].abs().max()
    else:
        optimal_threshold = None

    # calculate the distance between ground truth and predicted jacobian
    norm_diff: float = torch.norm(dep_mat.abs() - gt_jacobian_unmixing.abs()).mean()
    thresholded_norm_diff: float = torch.norm(
        dep_mat.abs() * (dep_mat.abs() > threshold) - gt_jacobian_unmixing.abs()
    ).mean()

    # calculate the fraction of correctly identified zeroes
    incorrect_edges: float = ((dep_mat.abs() * indirect_causes) > threshold).sum()
    sparsity_accuracy: float = 1.0 - incorrect_edges / (indirect_causes.sum() + 1e-8)

    metrics = JacobianMetrics(
        norm_diff,
        thresholded_norm_diff,
        optimal_threshold,
        sparsity_accuracy,
        amari_distance(dep_mat, gt_jacobian_mixing_permuted),
        permutation_loss(
            jacobian_to_tril_and_perm(dep_mat, qr=True)[0], matrix_power=True
        ),
    )

    return metrics


def jacobian_to_tril_and_perm(dep_mat, qr: bool = True):
    """
    The Jacobian of the learned network J should be W@P to invert the causal data generation process (SEM),
    where W is the inverse of the mixing matrix, and P is a permutation matrix
    (inverting the ordering back to its correct ordering).

    In this case, J:=WP (the data generation process has P^T@inv(W) ), and it should be (lower) triangular.
    Thus, we can proceed as follows:
    1. Calculate the Cholesky decomposition of JJ^T = WPP^TW^T = WW^T -> resulting in W
    2. Left-multiply J with W^-1 to get P

    :param dep_mat:
    :return:
    """
    if qr is True:
        Q, R = dep_mat.T.qr()
        inv_permutation = Q
        unmixing_tril_weight = R.T
    else:
        unmixing_tril_weight = (dep_mat @ dep_mat.T).cholesky()
        inv_permutation = unmixing_tril_weight.inverse() @ dep_mat

    return inv_permutation, unmixing_tril_weight


def check_permutation(candidate: torch.tensor, threshold: float = 0.95):
    hard_permutation = (candidate.abs() > threshold).float()
    success = permutation_loss(hard_permutation, False) == 0.0

    return hard_permutation, success.item()


from typing import Any, Dict, List, Optional, Tuple, Union

from torchmetrics.utilities.data import METRIC_EPS
from warnings import warn


class JacobianBinnedPrecisionRecall(Metric):
    """
    Based on https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/classification/binned_precision_recall.py#L45-L184
    """

    TPs: torch.Tensor
    FPs: torch.Tensor
    FNs: torch.Tensor

    def __init__(
        self,
        num_thresholds: Optional[int] = None,
        thresholds: Union[int, torch.Tensor, List[float], None] = None,
        log_base: Optional[float] = 10.0,
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(compute_on_step=compute_on_step, **kwargs)

        if isinstance(num_thresholds, int):
            self.num_thresholds = num_thresholds
            thresholds = torch.logspace(-5, 0, self.num_thresholds, base=log_base)
            self.register_buffer("thresholds", thresholds)
        elif thresholds is not None:
            if not isinstance(thresholds, (list, torch.Tensor)):
                raise ValueError(
                    "Expected argument `thresholds` to either be an integer, list of floats or a tensor"
                )
            thresholds = (
                torch.tensor(thresholds) if isinstance(thresholds, list) else thresholds
            )
            self.num_thresholds = thresholds.numel()
            self.register_buffer("thresholds", thresholds)

        for name in ("TPs", "FPs", "FNs"):
            self.add_state(
                name=name,
                default=torch.zeros(self.num_thresholds, dtype=torch.float32),
                dist_reduce_fx="sum",
            )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        """
        Args
            preds: (n_samples,) tensor
            target: (n_samples, ) tensor
        """
        preds, target = preds.reshape(-1,), target.reshape(
            -1,
        )

        assert preds.shape == target.shape

        if preds.min() < 0:
            preds = preds.abs()

        if (pred_max := preds.max()) != 1.0:
            preds /= pred_max

        target = target == 1
        # Iterate one threshold at a time to conserve memory
        for i in range(self.num_thresholds):
            predictions = preds >= self.thresholds[i]
            self.TPs[i] += (target & predictions).sum(dim=0)
            self.FPs[i] += ((~target) & (predictions)).sum(dim=0)
            self.FNs[i] += (target & (~predictions)).sum(dim=0)

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns float tensor of size n_classes."""
        precisions = (self.TPs + METRIC_EPS) / (self.TPs + self.FPs + METRIC_EPS)
        recalls = self.TPs / (self.TPs + self.FNs + METRIC_EPS)

        return precisions, recalls, self.thresholds
