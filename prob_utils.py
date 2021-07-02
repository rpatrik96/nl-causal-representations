import sys
from collections import Counter

import torch

from cl_ica import disentanglement_utils
from dep_mat import calc_jacobian


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
    (linear_disentanglement_score, _), _ = disentanglement_utils.linear_disentanglement(z, hz, mode="r2")
    (permutation_disentanglement_score, _,), _ = disentanglement_utils.permutation_disentanglement(
        z,
        hz,
        mode="pearson",
        solver="munkres",
        rescaling=True,
    )

    return linear_disentanglement_score, permutation_disentanglement_score


def check_independence_z_gz(ind_check, h_ind, latent_space):
    z_disentanglement = latent_space.sample_marginal(ind_check.hparams.n_eval_samples)
    lin_dis_score, perm_dis_score = calc_disentanglement_scores(z_disentanglement, h_ind(z_disentanglement))

    print(f"Id. Lin. Disentanglement: {lin_dis_score:.4f}")
    print(f"Id. Perm. Disentanglement: {perm_dis_score:.4f}")
    print('Run test with ground truth sources')

    if ind_check.hparams.use_dep_mat:
        # x \times z
        dep_mat = calc_jacobian(h_ind, z_disentanglement, normalize=ind_check.hparams.preserve_vol).abs().mean(0)
        print(dep_mat)
        null_list = [False] * torch.numel(dep_mat)
        null_list[torch.argmin(dep_mat).item()] = True
        var_map = [1, 1, 2, 2]
    else:
        null_list, var_map = ind_check.check_bivariate_dependence(h_ind(z_disentanglement), z_disentanglement)
    ######Note this is specific to a dense 2x2 triangular matrix!######
    if Counter(null_list) == Counter([False, False, False, True]):

        print('concluded a causal effect')

        for i, null in enumerate(null_list):
            if null:
                print('cause variable is X{}'.format(str(var_map[i])))

    else:
        print('no causal effect...?')
        sys.exit()