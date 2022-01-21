import itertools

import torch

from care_nl_ica.cl_ica import disentanglement_utils


def setup_marginal(args):
    device = args.device
    eta = torch.zeros(args.n)
    if args.space_type == "sphere":
        eta[0] = args.sphere_r

    if args.m_p == 1:
        sample_marginal = lambda space, size, device=device: space.laplace(
            eta, args.m_param, size, device
        )
    elif args.m_p == 2:
        sample_marginal = lambda space, size, device=device: space.normal(
            eta, args.m_param, size, device
        )
    elif args.m_p == 0:
        sample_marginal = lambda space, size, device=device: space.uniform(
            size, device=device
        )
    else:
        sample_marginal = (
            lambda space, size, device=device: space.generalized_normal(
                eta, args.m_param, p=args.m_p, size=size, device=device
            )
        )
    return sample_marginal


def sample_marginal_and_conditional(latent_space, size, device):
    z = latent_space.sample_marginal(size=size, device=device)
    z3 = latent_space.sample_marginal(size=size, device=device)
    z_tilde = latent_space.sample_conditional(z, size=size, device=device)

    return z, z_tilde, z3


def setup_conditional(args):
    device = args.device
    if args.c_p == 1:
        sample_conditional = lambda space, z, size, device=device: space.laplace(
            z, args.c_param, size, device
        )
    elif args.c_p == 2:
        sample_conditional = lambda space, z, size, device=device: space.normal(
            z, args.c_param, size, device
        )
    elif args.c_p == 0:
        sample_conditional = (
            lambda space, z, size, device=device: space.von_mises_fisher(
                z, args.c_param, size, device,
            )
        )
    else:
        sample_conditional = (
            lambda space, z, size, device=device: space.generalized_normal(
                z, args.c_param, p=args.c_p, size=size, device=device
            )
        )
    return sample_conditional


def laplace_log_cdf(x: torch.Tensor, signal_model: torch.distributions.laplace.Laplace):
    """
    Log cdf of the Laplace distribution (numerically stable).
    Source: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/internal/special_math.py#L281

    :param x: tensor to be transformed
    :param signal_model: Laplace distribution from torch.distributions.
    """

    # transform with location and scale
    x_tr = (x - signal_model.mean) / signal_model.scale

    # x < 0
    neg_res = torch.log(torch.tensor(0.5)) + x_tr

    # x >= 0
    pos_res = torch.log1p(-.5 * (-x_tr.abs()).exp())

    return torch.where(x < signal_model.mean, neg_res, pos_res)


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
    ksi_corr_mat = ksi_correlation(hz, z) + torch.tril(ksi_correlation(z, hz), -1)

    return DisentanglementMetrics(lin_score=lin_dis_score,
                                  perm_score=permutation_disentanglement_score,
                                  non_perm_score=non_perm_dis_score,

                                  perm_corr_mat=torch.tensor(perm_corr_mat),
                                  non_perm_corr_mat=torch.tensor(non_perm_corr_mat),
                                  ksi_corr_mat=ksi_corr_mat,

                                  perm_corr_diag=frobenius_diagonality(torch.tensor(perm_corr_mat)),
                                  non_perm_corr_diag=frobenius_diagonality(torch.tensor(non_perm_corr_mat)),
                                  ksi_corr_diag=frobenius_diagonality(ksi_corr_mat),

                                  perm_corr_mig=_mig_from_correlation(torch.tensor(perm_corr_mat)),
                                  non_perm_corr_mig=_mig_from_correlation(torch.tensor(non_perm_corr_mat)),
                                  ksi_corr_mig=_mig_from_correlation(ksi_corr_mat)
                                  )


from dataclasses import dataclass


@dataclass
class DisentanglementMetrics:
    lin_score: float
    perm_score: float
    non_perm_score: float

    perm_corr_mat: torch.Tensor
    non_perm_corr_mat: torch.Tensor
    ksi_corr_mat: torch.Tensor

    perm_corr_diag: float
    non_perm_corr_diag: float
    ksi_corr_diag: float

    perm_corr_mig: float
    non_perm_corr_mig: float
    ksi_corr_mig: float


def _mig_from_correlation(corr: torch.Tensor):
    off_diag_abs = (corr - corr.diag().diag()).abs()

    return (corr.abs().diag() - off_diag_abs.max(0)[0]).mean()


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


def frobenius_diagonality(matrix: torch.Tensor) -> float:
    """
    Calculates the IMA constrast (the lefy KL measure of diagonality).

    :param matrix: matrix as a torch.Tensor
    :return:
    """

    # this is NOT IMA CONTRAST, BUT A DIAGONALITY MEASURE

    # matrix is here a correlation matrix (to yield the modified Frobenius diagonality measure of https://www.sciencedirect.com/science/article/pii/S0024379516303834#se0180)
    return .5 * ((matrix - torch.eye(matrix.shape[0], device=matrix.device)).norm('fro').pow(2)).log().item()

    return 0.5 * (torch.linalg.slogdet(torch.diag(torch.diag(matrix)))[1] -
                  torch.linalg.slogdet(matrix)[1]).item()


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
