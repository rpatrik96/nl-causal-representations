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
    def __init__(self, num_dim: int, num_layers: int, num_samples: int):
        """
        :param num_dim: number of dimensions
        :param  num_layers: number of layers
        :param num_samples: number of samples
        """
        super().__init__()

        self.num_dim = num_dim
        self.num_layers = num_layers
        self.num_samples = num_samples

        self.linears = nn.ModuleList([
            nn.Linear(self.num_dim, self.num_dim, bias=False)
            for _ in range(self.num_layers)
        ])

        # make it lower triangular
        for i in range(self.num_layers - 1):
            self.linears[i].weight.data = torch.tril(self.linears[i].weight.data)

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
