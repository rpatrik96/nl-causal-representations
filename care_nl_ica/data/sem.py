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
        diag_weight: float = 0.0,
        offset: float = 1.0,
        mask_prob=1.0,
        weight_rand_func="rand",
    ):
        super().__init__()
        self.variant = variant
        self.num_vars = num_vars

        if weight_rand_func == "rand":
            weight_rand_func = torch.rand
        elif weight_rand_func == "randn":
            weight_rand_func = torch.randn
        else:
            raise ValueError

        if force_chain is False and permute is True:
            print(f"-----As {permute=}, setting force_chain=True")
            force_chain = True

        # weight init
        inv_weight = torch.tril(
            # torch.randn((num_vars, num_vars)).tril() + diag_weight * torch.eye(num_vars)
            weight_rand_func((num_vars, num_vars)).tril()
            + offset * torch.ones((num_vars, num_vars))
            + diag_weight * torch.eye(num_vars)
        )

        if force_uniform is True:
            print("---------Forcing uniform weights---------")
            inv_weight = torch.tril(torch.ones(num_vars, num_vars))

        # construct a chain
        if force_chain is True or mask_prob != 0.0:
            mask = torch.tril(torch.ones_like(inv_weight))

            zeros_in_chain = torch.tril(torch.ones_like(inv_weight), -2)
            mask[zeros_in_chain == 1] = 0

            if mask_prob != 1.0:
                mask = (
                    (
                        mask
                        + torch.tril(
                            torch.bernoulli(mask_prob * torch.ones_like(inv_weight)), 1
                        )
                    )
                    .bool()
                    .float()
                )

            inv_weight *= mask

        print(f"{inv_weight=}")

        self.weight = inv_weight.inverse().tril()
        print(f"{self.weight=}")

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
        return self.permutation((self.weight @ x.T).T)

    def to(self, device):
        """
        Move the model to the specified device.

        :param device: The device to move the model to.
        """
        super().to(device)
        self.weight = self.weight.to(device)
        # self.mask = self.mask.to(device)

        return self


class NonLinearSEM(LinearSEM):
    def __init__(
        self,
        num_vars: int,
        permute: bool = False,
        variant=-1,
        force_chain: bool = False,
        force_uniform: bool = False,
        diag_weight: float = 0.0,
        offset: float = 1.0,
        mask_prob=0.0,
        weight_rand_func="rand",
    ):
        super().__init__(
            num_vars=num_vars,
            permute=permute,
            variant=variant,
            force_chain=force_chain,
            force_uniform=force_uniform,
            diag_weight=diag_weight,
            offset=offset,
            mask_prob=mask_prob,
            weight_rand_func=weight_rand_func,
        )

        self.slopes = torch.rand(num_vars).clip(0.25, 1)
        print(f"{self.slopes=}")
        print("-------fixing slopes to 0.25--------")
        self.relus = [
            lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.25)
            for s in self.slopes
        ]

    def forward(self, x):
        # z = torch.zeros_like(x)
        # w = self.weight

        # mix  =
        #
        # for i in range(self.num_vars):
        #     z[:,i] = self.relus[i]
        # if i != 0:
        #     z[:, i] = self.relus[i](w[i, i] * x[:, i] + z[:, :i] @ w[i, :i])
        # else:
        #     z[:, i] = w[i, i] * self.relus[i](x[:, i])

        return self.permutation(self.relus[0]((self.weight @ x.T).T))
