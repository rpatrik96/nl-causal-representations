import itertools
from dataclasses import dataclass

import torch

import disentanglement_utils


@dataclass
class JacobianMetrics:
    norm_diff:float
    thresholded_norm_diff:float
    optimal_threshold:float
    sparsity_accuracy:float
    amari_distance:float
    permutation_quality:float

@dataclass
class DisentanglementMetrics:
    lin_score: float = 0
    perm_score: float = 0
    non_perm_score: float = 0

    perm_corr_mat: torch.Tensor = None
    non_perm_corr_mat: torch.Tensor = None
    ksi_corr_mat: torch.Tensor = None

    perm_corr_diag: float = 0
    non_perm_corr_diag: float = 0
    ksi_corr_diag: float = 0

    perm_corr_mig: float = 0
    non_perm_corr_mig: float = 0
    ksi_corr_mig: float = 0



def _mig_from_correlation(corr: torch.Tensor):
    sorted_corr = torch.sort(corr.abs(), dim=1, descending=True)[0]
    return (sorted_corr[:, 0] - sorted_corr[:, 1]).abs().mean()


def amari_distance(W: torch.Tensor, A: torch.Tensor) -> float:
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

    P = W @ A

    def s(r):
        return ((r ** 2).sum(axis=1) / (r ** 2).max(axis=1)[0] - 1).sum()

    return ((s(P.abs()) + s(P.T.abs())) / (2 * P.shape[1])).item()


def frobenius_diagonality(matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Frobenius measure of diagonality for correlation matrices.
    Source: https://www.sciencedirect.com/science/article/pii/S0024379516303834#se0180

    :param matrix: matrix as a torch.Tensor
    :return:
    """

    return .5 * ((matrix - torch.eye(matrix.shape[0], device=matrix.device)).norm('fro').pow(2))


def corr_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """

    :param x: torch.Tensor of size (dim, num_samples)
    :param y: torch.Tensor of size (dim, num_samples)
    :return:
    """

    dim = x.shape[0]

    corr_mat = torch.zeros(dim, dim, device=x.device)

    for i in range(dim):
        for j in range(dim):
            corr_mat[i, j] = torch.corrcoef(torch.stack((x[i, :], y[j, :])))[1, 0]

    return corr_mat


def ksi_correlation(hz: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Calculates the correlation between the latent variables and the observed variables.
    :param hz: latent variables
    :param z: observed variables
    """
    num_samples = z.shape[0]
    num_dim = z.shape[1]

    combinations = list(itertools.combinations_with_replacement(list(range(num_dim)), 2))

    # from http://arxiv.org/abs/1909.10140
    # 1. take the (zi, hzi) pairs (for each dimension),
    # sort zi and
    # use the indices that sort zi to sort hzi in ascending order
    sorted_representations = [hz[:, j][torch.sort(z[:, i], axis=-1)[1]] for (i, j) in combinations]
    # 2. rank the sorted sorted_representations dimensionwise (i.e.,s_repr),
    # where the rank of each item is the number of hzi_sorted s.t.
    # it counts the smaller elements that item
    representation_ranks = [torch.tensor([(s_repr <= item).sum() for item in s_repr]) for s_repr in
                            sorted_representations]
    # 3. use eq. 11  (assumes no ties - ties can be ignored for large num_samples)
    ksi = [1 - 3 * (r[1:] - r[:-1]).abs().sum() / (num_samples ** 2 - 1) for r in representation_ranks]

    # arrange into matrix - note that it is not symmetric
    ksi_matrix = torch.zeros(num_dim, num_dim, device=z.device)
    for idx, (i, j) in enumerate(combinations):
        ksi_matrix[i, j] = ksi[idx]

    # +1: normalize by the possible min and max values
    ksi_max = (num_samples - 2) / (num_samples + 1)
    ksi_min = -.5 + 1 / num_samples

    return ksi_matrix




def extract_permutation_from_jacobian(dep_mat, qr:bool=False):
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


def permutation_loss(matrix: torch.Tensor, matrix_power: bool = False):
    if matrix_power is False:
        # rows and cols sum up to 1
        col_sum = matrix.abs().sum(0)
        row_sum = matrix.abs().sum(1)
        loss = ((col_sum - torch.ones_like(col_sum)).norm() + (
                row_sum - torch.ones_like(row_sum)).norm())
    else:
        # diagonality (as Q^n = I for permutation matrices)
        loss = frobenius_diagonality(matrix.matrix_power(matrix.shape[0]).abs())

    return loss


def calc_disentanglement_scores(z, hz):
    (lin_dis_score, _), _ = disentanglement_utils.linear_disentanglement(z, hz, mode="r2")
    (permutation_disentanglement_score, perm_corr_mat,), _ = disentanglement_utils.permutation_disentanglement(
        z,
        hz,
        mode="pearson",
        solver="munkres",
        rescaling=True,
    )

    # rescale variables
    assert z.shape == hz.shape
    # find beta_j that solve Y_ij = X_ij beta_j

    beta = torch.diag((z.detach() * hz.detach()).sum(0) / (hz.detach() ** 2).sum(0))

    non_perm_dis_score, non_perm_corr_mat = disentanglement_utils._disentanglement(z.detach().cpu().numpy(),
                                                                                   (hz.detach() @ beta).cpu().numpy(),
                                                                                   mode="pearson",
                                                                                   reorder=False)

    # the metrics is not symmetric
    # and we don't need the diagonal twice
    ksi_corr_mat = ksi_correlation(hz.detach(), z.detach()) + torch.tril(ksi_correlation(z.detach(), hz.detach()), -1)

    return DisentanglementMetrics(lin_score=lin_dis_score,
                                  perm_score=permutation_disentanglement_score,
                                  non_perm_score=non_perm_dis_score,

                                  perm_corr_mat=torch.tensor(perm_corr_mat),
                                  non_perm_corr_mat=torch.tensor(non_perm_corr_mat),
                                  ksi_corr_mat=ksi_corr_mat,

                                  perm_corr_diag=frobenius_diagonality(torch.tensor(perm_corr_mat)).item(),
                                  non_perm_corr_diag=frobenius_diagonality(torch.tensor(non_perm_corr_mat)).item(),
                                  ksi_corr_diag=frobenius_diagonality(ksi_corr_mat).item(),

                                  perm_corr_mig=_mig_from_correlation(torch.tensor(perm_corr_mat)),
                                  non_perm_corr_mig=_mig_from_correlation(torch.tensor(non_perm_corr_mat)),
                                  ksi_corr_mig=_mig_from_correlation(ksi_corr_mat)
                                  )