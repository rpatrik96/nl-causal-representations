import torch


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
