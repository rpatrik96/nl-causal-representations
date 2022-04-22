import itertools
import math

import torch
from torch import nn as nn


class LinearSEM(nn.Module):
    def __init__(
        self,
        num_vars: int,
        permute: bool = False,
        variant: int = -1,
        force_chain: bool = False,
        force_uniform: bool = False,
    ):
        super().__init__()
        self.variant = variant
        self.num_vars = num_vars

        # weight init
        self.weight = nn.Parameter(torch.tril(nn.Linear(num_vars, num_vars).weight))
        if force_uniform is True:
            print("---------Forcing uniform weights---------")
            self.weight = nn.Parameter(torch.tril(torch.ones(num_vars, num_vars)))
        print(f"{self.weight=}")

        self.mask = (
            (
                torch.tril(torch.bernoulli(1.0 * torch.ones_like(self.weight)), 1)
                + torch.eye(num_vars)
            )
            .bool()
            .float()
        )

        # construct a chain
        if force_chain is True:
            print("-------Forcing chain-------")
            self.mask = torch.tril(torch.ones_like(self.weight))

            zeros_in_chain = torch.tril(torch.ones_like(self.weight), -2)
            self.mask[zeros_in_chain == 1] = 0
        # else:
        #     # ensure that the first column is not masked,
        #     # so the causal ordering will be unique 0-> this is not enough
        #     self.mask[:, 0] = 1

        self.mask.requires_grad = False
        print(f"{self.mask=}")

        self._setup_permutation(permute)

    def _setup_permutation(self, permute):

        if self.variant == -1:
            self.permute_indices = torch.randperm(self.num_vars)
        else:
            if self.variant < (fac := math.factorial(self.num_vars)):
                self.permute_indices = torch.tensor(
                    list(
                        itertools.islice(
                            itertools.permutations(range(self.num_vars)),
                            self.variant,
                            self.variant + 1,
                        )
                    )[0]
                )
            else:
                raise ValueError(f"{self.variant=} should be smaller than {fac}")

        self.permutation = (
            (lambda x: x)
            if permute is False
            else (lambda x: x[:, self.permute_indices])
        )

        print(f"{self.permute_indices=}")

    @property
    def permutation_matrix(self) -> torch.Tensor:
        m = torch.zeros_like(self.weight)
        m[list(range(self.num_vars)), self.permute_indices] = 1

        return m

    def forward(self, x):
        z = torch.zeros_like(x)
        w = torch.tril(self.weight * self.mask)

        for i in range(self.num_vars):
            z[:, i] = w[i, i] * x[:, i]

            if i != 0:
                z[:, i] = z[:, i] + z[:, :i] @ w[i, :i]

        return self.permutation(z)

    def to(self, device):
        """
        Move the model to the specified device.

        :param device: The device to move the model to.
        """
        super().to(device)
        self.weight = self.weight.to(device)
        self.mask = self.mask.to(device)

        return self


class NonLinearSEM(LinearSEM):
    def __init__(
        self,
        num_vars: int,
        permute: bool = False,
        variant=-1,
        force_chain: bool = False,
        force_uniform: bool = False,
    ):
        super().__init__(
            num_vars=num_vars,
            permute=permute,
            variant=variant,
            force_chain=force_chain,
            force_uniform=force_uniform,
        )

        self.relus = [
            lambda x: torch.nn.functional.leaky_relu(x, negative_slope=s)
            for s in torch.rand(num_vars).clip(0.1, 1)
        ]

    def forward(self, x):

        z = torch.zeros_like(x)
        w = torch.tril(self.weight * self.mask)

        for i in range(self.num_vars):
            if i != 0:
                z[:, i] = self.relus[i](w[i, i] * x[:, i] + z[:, :i] @ w[i, :i])
            else:
                z[:, i] = w[i, i] * self.relus[i](x[:, i])

        return self.permutation(z)
