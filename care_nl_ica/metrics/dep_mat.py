from dataclasses import dataclass

import torch

from care_nl_ica.losses.dep_mat import permutation_loss
from care_nl_ica.metrics.ica_dis import amari_distance

from typing import Dict


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
    gt_jacobian_encoder,
    indirect_causes,
    gt_jacobian_decoder_permuted,
    threshold: float = 1e-3,
) -> JacobianMetrics:
    # calculate the optimal threshold for 1 accuracy
    # calculate the indices where the GT is 0 (in the lower triangular part)
    sparsity_mask = (torch.tril(gt_jacobian_encoder.abs() < 1e-6)).bool()

    if sparsity_mask.sum() > 0:
        optimal_threshold = dep_mat[sparsity_mask].abs().max()
    else:
        optimal_threshold = None

    # calculate the distance between ground truth and predicted jacobian
    norm_diff: float = torch.norm(dep_mat.abs() - gt_jacobian_encoder.abs()).mean()
    thresholded_norm_diff: float = torch.norm(
        dep_mat.abs() * (dep_mat.abs() > threshold) - gt_jacobian_encoder.abs()
    ).mean()

    # calculate the fraction of correctly identified zeroes
    incorrect_edges: float = ((dep_mat.abs() * indirect_causes) > threshold).sum()
    sparsity_accuracy: float = 1.0 - incorrect_edges / (indirect_causes.sum() + 1e-8)

    metrics = JacobianMetrics(
        norm_diff,
        thresholded_norm_diff,
        optimal_threshold,
        sparsity_accuracy,
        amari_distance(dep_mat, gt_jacobian_decoder_permuted),
        permutation_loss(
            extract_permutation_from_jacobian(dep_mat, qr=True), matrix_power=True
        ),
    )

    return metrics


def extract_permutation_from_jacobian(dep_mat, qr: bool = False):
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
        permutation_estimate = dep_mat.T.qr()[0].T
    else:
        unmixing_tril = (dep_mat @ dep_mat.T).cholesky()
        permutation_estimate = unmixing_tril.inverse() @ dep_mat

    return permutation_estimate


def check_permutation(candidate: torch.tensor, threshold: float = 0.95):
    hard_permutation = (candidate.abs() > threshold).float()
    success = permutation_loss(hard_permutation, False) == 0.0

    return hard_permutation, success.item()
