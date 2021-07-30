import torch
import torch.nn as nn


class ICAModel(nn.Module):
    """
    Linear ICA class
    """

    def __init__(self, dim: int, signal_model: torch.distributions.distribution.Distribution):
        """
        :param dim: an integer specifying the number of signals
        :param signal_model: class of the signal model distribution
        """
        super().__init__()

        self.W = torch.nn.Parameter(torch.eye(dim))
        self.signal_model = signal_model

    def forward(self, x: torch.Tensor):
        """
        :param x : data tensor of (num_samples, signal_dim)
        """

        # unmixing
        return torch.matmul(x, self.W)

    @staticmethod
    def _ml_objective(x: torch.Tensor, signal_model: torch.distributions.laplace.Laplace):
        """
        Implements the ML objective

        :param x: tensor to be transformed
        :param signal_model: Laplace distribution from torch.distributions.
        """

        # transform with location and scale
        x_tr = (x - signal_model.mean).abs() / signal_model.scale

        return -x_tr - torch.log(2 * signal_model.scale)

    def loss(self, x: torch.Tensor):
        """
        :param x : data tensor of (num_samples, signal_dim)
        """

        # ML objective
        model_entropy = self._ml_objective(self(x), self.signal_model).mean()

        # log of the abs determinant of the unmixing matrix
        # _, log_abs_det = torch.linalg.slogdet(self.W)
        _, log_abs_det = torch.slogdet(self.W)

        # as we need to minimize, there is a minus sign here
        return -model_entropy, -log_abs_det

    def dependency(self, row_idx: int, col_idx: int):
        return self.W[row_idx, col_idx]
