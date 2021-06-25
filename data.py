from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class LinearDataset(Dataset):
    def __init__(self, a: float, b: float, c: float, num_samples: int):
        """
        :param a: scalar parameter for Z1 in Z2
        :param b: scalar parameter for Z2 in Z3
        :param c: scalar parameter for Z1 in Z3
        :param num_samples: number of samples
        """
        super().__init__()

        self.a = a
        self.b = b
        self.c = c
        self.num_samples = num_samples

        self.data = self._generate_data()

    def _generate_data(self) -> torch.Tensor:
        """
        Generates  data for the causal graph with the adjacency matrix of
             1  0  0
        A = -a  1  0
            -c -b  1
        """
        noise_dist = torch.distributions.Laplace(0, 1)

        # Z1
        N1 = noise_dist.sample((self.num_samples,))
        Z1 = N1

        # Z2
        N2 = noise_dist.sample((self.num_samples,))
        Z2 = self.a * Z1 + N2

        # Z3
        N3 = noise_dist.sample((self.num_samples,))
        Z3 = self.b * Z2 + self.c * Z1 + N3

        self.noise = torch.stack((N1, N2, N3))

        return torch.stack((Z1, Z2, Z3), dim=1)

    def __getitem__(self, i):
        return self.data[i, :]

    def __len__(self):
        return self.num_samples


class NonLinearDataset(Dataset):
    def __init__(self, num_dim: int, num_layers: int, num_samples: int, variant:int=None):
        """
        :param variant:
        :param num_dim: number of dimensions
        :param  num_layers: number of layers
        :param num_samples: number of samples
        """
        super().__init__()

        if variant is not None and variant > (max_elem:=num_dim * (num_dim - 1) // 2):
            raise ValueError(f"{variant=}, should be lower than {max_elem}!")

        self.num_dim = num_dim
        self.num_layers = num_layers
        self.num_samples = num_samples
        self.variant = variant if variant is None else torch.IntTensor([variant])

        self.graph_idx, self.tril_mask = createARmask(self.num_dim, variant)

        self.linears = nn.ModuleList([
            nn.Linear(self.num_dim, self.num_dim, bias=False)
            for _ in range(self.num_layers)
        ])

        # make it lower triangular
        for i in range(self.num_layers - 1):
            self.linears[i].weight.data = self.tril_mask * self.linears[i].weight.data

        self.relus = nn.ModuleList([
            nn.LeakyReLU()
            for _ in range(self.num_layers - 1)
        ])

        self.data = self._generate_data()

    def _generate_data(self) -> torch.Tensor:
        """
        Generates  data for the causal graph according to the Monti et al. paper

        """
        noise_dist = torch.distributions.Laplace(0, 1)
        self.noise = noise_dist.sample((self.num_samples, self.num_dim))

        X = self.linears[0](self.noise)
        for i in range(1, self.num_layers):
            X = self.linears[i](self.relus[i - 1](X))

        return X.detach()

    def __getitem__(self, i):
        return self.data[i, :]

    def __len__(self):
        return self.num_samples

    @property
    def num_edges(self) -> torch.Tensor:
        """
        Number of edges in the AR graph
        :return: number of edges
        """
        return self.tril_mask.count_nonzero()


def tensor2bitlist(x: torch.Intensor, bits: int) -> torch.Tensor:
    """
    Converts an integer into a list of binary tensors.

    SourceÉ https://stackoverflow.com/a/63546308

    :param x: number to convert into binary
    :param bits: number of bits to use
    :return:
    """

    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def createARmask(dim: int, variant:torch.IntTensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a (sparse) autoregressive triangular mask

    :param dim: dimensionality of the matrix
    :return:
    """

    # constants
    mask_numel = dim * (dim - 1) // 2
    row_idx, col_idx = torch.tril_indices(dim, dim, -1)

    if variant is None:
        max_variants = 2 ** mask_numel
        variant = torch.randint(max_variants, (1,)).int()

    # create mask elements
    mask_elem = tensor2bitlist(variant, mask_numel)

    # fill the mask
    mask = torch.eye(dim)
    mask[row_idx, col_idx] = mask_elem

    return variant, mask
