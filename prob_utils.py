import torch



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