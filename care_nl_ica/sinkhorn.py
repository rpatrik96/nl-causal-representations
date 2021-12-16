import torch
from torch import nn as nn


class SinkhornOperator(object):
    def __init__(self, num_steps: int):

        if num_steps < 1:
            raise ValueError(f"{num_steps=} should be at least 1")

        self.num_steps = num_steps

    def __call__(self, matrix: torch.Tensor) -> torch.Tensor:


        def _normalize_row(matrix: torch.Tensor) -> torch.Tensor:
            return matrix - torch.logsumexp(matrix, 1, keepdim=True)

        def _normalize_column(matrix: torch.Tensor) -> torch.Tensor:
            return matrix - torch.logsumexp(matrix, 0, keepdim=True)

        S = matrix

        for _ in range(self.num_steps):
            S = _normalize_column(_normalize_row(S))

        return torch.exp(S)


class SinkhornNet(nn.Module):
    def __init__(self, num_dim: int, num_steps: int, temperature: float = 1):
        super().__init__()

        self.temperature = temperature

        self.sinkhorn_operator = SinkhornOperator(num_steps)
        # self.weight = nn.Parameter(nn.Linear(num_dim, num_dim).weight+0.5*torch.ones(num_dim,num_dim), requires_grad=True)
        self.weight = nn.Parameter(torch.ones(num_dim, num_dim), requires_grad=True)


    @property
    def doubly_stochastic_matrix(self) -> torch.Tensor:

        eps = 1e-10
        u = torch.empty_like(self.weight).uniform_(0,1)
        gumbel = -torch.log(-torch.log(u+eps)+eps)
        

        return self.sinkhorn_operator((self.weight+0.001*gumbel) / self.temperature)

    def forward(self, x) -> torch.Tensor:
        return self.doubly_stochastic_matrix @ x

    def to(self, device):
        """
        Move the model to the specified device.

        :param device: The device to move the model to.
        """
        super().to(device)
        self.weight = self.weight.to(device)

        return self


class DoublyStochasticMatrix(nn.Module):
    def __init__(self, num_vars: int, temperature: float = 1.):
        super().__init__()

        self.temperature = temperature
        self.num_vars = num_vars
        self.weight = nn.Parameter(nn.Linear(num_vars - 1, num_vars - 1).weight)

    @property
    def matrix(self):
        beta = torch.sigmoid(self.weight / self.temperature)

        l = ...
        u = ...
        x = l + beta * (u - l)

        return x
