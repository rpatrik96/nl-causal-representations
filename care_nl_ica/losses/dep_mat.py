import torch

from care_nl_ica.metrics.ica_dis import frobenius_diagonality


def sparsity_loss(dep_mat: torch.Tensor) -> torch.Tensor:
    """
    Calculates the sparsity-inducing (i.e., L1) loss for the dependency matrix.

    :param dep_mat: dependency matrix as a torch.Tensor
    :return: the sparsity loss as a torch.Tensor (scalar)
    """
    return dep_mat.abs().sum()


def triangularity_loss(dep_mat: torch.Tensor) -> torch.Tensor:
    """
    Calculates the loss term for inducing a **lower** triangular structure for the dependency matrix.
    This is calculated as the L1 norm of the upper triangular part of the dependency matrix
    (except the main diagonal).

    :param dep_mat: dependency matrix as a torch.Tensor
    :return: the triangularity loss as a torch.Tensor (scalar)
    """

    return torch.triu(dep_mat, 1).abs().mean()


def dependency_loss(
    dep_mat: torch.Tensor, weight_sparse: float = 1.0, weight_triangular: float = 1.0
) -> torch.Tensor:
    """
    Calculates the weighted sum of the triangularity-enforcing and the sparsity-inducing losses for the
    dependency matrix.

    :param dep_mat: dependency matrix as a torch.Tensor
    :param weight_sparse: scalar for weighting the sparsity loss
    :param weight_triangular: scalar for weighting the triangularity loss
    :return: sum of the two losses as a torch.Tensor
    """

    sparse_loss = sparsity_loss(dep_mat)
    triangular_loss = triangularity_loss(dep_mat)

    return weight_sparse * sparse_loss + weight_triangular * triangular_loss


def permutation_loss(matrix: torch.Tensor, matrix_power: bool = False):
    if matrix_power is False:
        # rows and cols sum up to 1
        col_sum = matrix.abs().sum(0)
        row_sum = matrix.abs().sum(1)
        loss = (col_sum - torch.ones_like(col_sum)).pow(2).mean() + (
            row_sum - torch.ones_like(row_sum)
        ).pow(2).mean()
    else:
        # diagonality (as Q^n = I for permutation matrices)
        loss = frobenius_diagonality(matrix.matrix_power(matrix.shape[0]).abs())

    return loss
