import torch
from torch import nn as nn


class SinkhornOperator(object):
    """
    From http://arxiv.org/abs/1802.08665
    """

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

        self.num_dim = num_dim
        self.temperature = temperature

        self.sinkhorn_operator = SinkhornOperator(num_steps)
        # self.weight = nn.Parameter(nn.Linear(num_dim, num_dim).weight+0.5*torch.ones(num_dim,num_dim), requires_grad=True)
        self.weight = nn.Parameter(torch.ones(num_dim, num_dim), requires_grad=True)

    @property
    def doubly_stochastic_matrix(self) -> torch.Tensor:
        return self.sinkhorn_operator(self.weight / self.temperature)

    def forward(self, x) -> torch.Tensor:
        if (dim_idx := x.shape.index(self.num_dim)) == 0 or len(x.shape) == 3:
            return self.doubly_stochastic_matrix @ x
        elif dim_idx == 1:
            return x @ self.doubly_stochastic_matrix

    def to(self, device):
        """
        Move the model to the specified device.

        :param device: The device to move the model to.
        """
        super().to(device)
        self.weight = self.weight.to(device)

        return self


if __name__ == "__main__":
    NUM_DIM = 3
    s = SinkhornNet(NUM_DIM, 20, 1e-4)
    s2 = SinkhornNet(NUM_DIM, 20, 1e-4)
    optim = torch.optim.Adam(list(s.parameters()) + list(s2.parameters()), lr=1e-4)
    permute_indices = [2, 0, 1]

    permute_mat = torch.zeros(NUM_DIM, NUM_DIM)
    permute_mat[list(range(NUM_DIM)), permute_indices] = 1

    # generate chain
    # weight = torch.tril(torch.ones_like(permute_mat), -2)
    # mask = torch.tril(torch.ones_like(weight))
    # zeros_in_chain = torch.tril(torch.ones_like(weight), -2)
    # mask[zeros_in_chain == 1] = 0

    # J
    # mixing = torch.tril(torch.randn(NUM_DIM, NUM_DIM)) * mask
    # J_permuted = mixing @ permute_mat
    J_permuted = torch.tensor(
        [
            [-1.8755, 0.00046791, 0.734563],
            [0.0003417, 0.73327672, -1.2309],
            [-0.00432, 1.8387, 0.00018102],
        ]
    )
    print(f"{J_permuted=}")

    for i in range(6000):

        optim.zero_grad()
        matrix = (
            s2.doubly_stochastic_matrix
            @ J_permuted.float()
            @ s.doubly_stochastic_matrix
        )
        # loss = 10*torch.triu(matrix, 1).abs().sum() #
        loss = -torch.tril(matrix, 0).abs().sum() * 10

        loss.backward()

        if i % 1000 == 0:
            print(f"{loss.item():.3f}")
        #     print(s.doubly_stochastic_matrix.detach())

        optim.step()

    # print(f"{permute_mat.T=}")
    print(f"S={s.doubly_stochastic_matrix.detach()}")
    print(f"S2={s2.doubly_stochastic_matrix.detach()}")
    print(s2.doubly_stochastic_matrix @ J_permuted.float() @ s.doubly_stochastic_matrix)
