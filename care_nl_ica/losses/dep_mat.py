import torch

from care_nl_ica.metrics.ica_dis import frobenius_diagonality


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
