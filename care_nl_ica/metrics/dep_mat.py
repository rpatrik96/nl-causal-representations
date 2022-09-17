from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import METRIC_EPS


def correct_jacobian_permutations(
    dep_mat: torch.Tensor, ica_permutation: torch.Tensor, sem_permutation: torch.Tensor
) -> torch.Tensor:
    return ica_permutation @ dep_mat @ sem_permutation


def correct_ica_scale_permutation(
    dep_mat: torch.Tensor,
    permutation: torch.Tensor,
    gt_jacobian_unmixing: torch.Tensor,
    hsic_adj,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    :param hsic_adj: adjacency matrix estimated by HSIC
    :param permutation: permutation matrix
    :param dep_mat: estimated unmixing Jacobian (including ICA permutation indeterminacy, i.e., P@J_GT)
    :param gt_jacobian_unmixing:
    :return:
    """
    # permutation = torch.eye(dep_mat.shape[0])
    scaled_appr_permutation_est_inv: torch.Tensor = (
        (dep_mat @ permutation @ gt_jacobian_unmixing.inverse()).inverse().contiguous()
    )
    dim = dep_mat.shape[0]
    num_zeros = dim**2 - dim
    zero_idx = (
        scaled_appr_permutation_est_inv.abs()
        .view(
            -1,
        )
        .sort()[1][:num_zeros]
    )

    # print(scaled_appr_permutation_est_inv)
    # sys.exit(0)

    # zero them out
    scaled_appr_permutation_est_inv.view(-1, 1)[zero_idx] = 0

    # torch.linalg.solve()
    # print(scaled_appr_permutation_est_inv)
    print(scaled_appr_permutation_est_inv.abs())
    return (
        scaled_appr_permutation_est_inv @ dep_mat @ permutation,
        None
        if hsic_adj is None
        else scaled_appr_permutation_est_inv.abs().bool().float()
        @ hsic_adj
        @ permutation,
    )


def jacobian_edge_accuracy(
    dep_mat: torch.Tensor, gt_jacobian_unmixing: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the accuracy of detecting edges based on the GT Jacobian and the estimated one such that the smallest N
    absolute elements of the estimated `dep_mat` are set to 0, where N is the number of zeros in `gt_jacobian_unmixing`
    :param dep_mat:
    :param gt_jacobian_unmixing:
    :return:
    """
    num_zeros = (gt_jacobian_unmixing == 0.0).sum()

    # query indices of smallest absolute values
    zero_idx = (
        dep_mat.abs()
        .view(
            -1,
        )
        .sort()[1][:num_zeros]
    )

    # zero them out
    dep_mat.view(
        -1,
    )[zero_idx] = 0

    return (dep_mat.bool() == gt_jacobian_unmixing.bool()).float().mean()


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
        start=-4,
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(compute_on_step=compute_on_step, **kwargs)

        if isinstance(num_thresholds, int):
            self.num_thresholds = num_thresholds
            if log_base > 1:
                thresholds = torch.logspace(
                    start, 0, self.num_thresholds, base=log_base
                )
            else:
                thresholds = torch.linspace(10**start, 1.0, num_thresholds)
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
        preds, target = (
            preds.reshape(
                -1,
            ).abs(),
            target.reshape(
                -1,
            ).abs(),
        )

        assert preds.shape == target.shape

        if (pred_max := preds.max()) != 1.0:
            preds /= pred_max

        target = target.bool()
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
