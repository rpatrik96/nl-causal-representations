import torch
from torch import nn as nn
from care_nl_ica.losses.dep_mat import permutation_loss


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


from scipy.spatial.distance import hamming


def learn_permutation(
    true_jac,
    est_jac,
    permute_indices,
    num_steps=3000,
    tril_weight=10.0,
    triu_weigth=10.0,
    diag_weight=0.0,
    lr=1e-3,
    verbose=False,
    drop_smallest=False,
    threshold=None,
    binary=False,
    hamming_threshold=2e-2,
    dag_permute: bool = True,
    eps=1e-8,
    row_normalize: bool = False,
    rank_acc=False,
):
    est_jac = torch.from_numpy(est_jac).float()
    true_jac = torch.from_numpy(true_jac).float()

    if row_normalize is True:
        est_jac /= est_jac.norm(dim=1).reshape(-1, 1)
    # print(est_jac)
    dim = est_jac.shape[0]
    j_hamming = lambda gt, est: hamming(
        gt.abs().reshape(
            -1,
        )
        > hamming_threshold,
        est.detach()
        .abs()
        .reshape(
            -1,
        )
        > hamming_threshold,
    )
    j_acc = (
        lambda gt, est: (
            (
                gt.abs().reshape(
                    -1,
                )
                > eps
            )
            == (
                est.detach()
                .abs()
                .reshape(
                    -1,
                )
                > hamming_threshold
            )
        )
        .float()
        .mean()
    )

    if drop_smallest is True:
        zero_idx = (
            est_jac.abs()
            .view(
                -1,
            )
            .sort()[1][: dim * (dim - 1) // 2]
        )
        est_jac.view(
            -1,
        )[zero_idx] = 0

    if threshold is not None:
        est_jac[est_jac.abs() < threshold] = 0

    if binary is True:
        est_jac = est_jac.bool().float()
    s_dag = SinkhornNet(dim, 20, 1e-4)
    s_ica = SinkhornNet(dim, 20, 1e-4)
    if dag_permute is True:
        optim = torch.optim.Adam(
            list(s_dag.parameters()) + list(s_ica.parameters()), lr=lr
        )
    else:
        optim = torch.optim.Adam(s_ica.parameters(), lr=lr)
    for i in range(num_steps):

        optim.zero_grad()
        if dag_permute is True:
            matrix = (
                s_ica.doubly_stochastic_matrix
                @ est_jac.abs()
                @ s_dag.doubly_stochastic_matrix
            )
        else:
            matrix = s_ica.doubly_stochastic_matrix @ est_jac.abs()
        loss_l = -tril_weight * torch.tril(matrix, 0).abs().sum()
        loss_u = triu_weigth * torch.triu(matrix, 1).abs().sum()
        loss_diag = diag_weight * (1.0 / (matrix.diag() + eps)).sum()

        loss = loss_l + loss_u + loss_diag

        loss.backward()

        if i % 250 == 0 and dag_permute is True:
            correct_order = torch.all(
                s_dag.doubly_stochastic_matrix.max(1)[1]
                == torch.tensor(permute_indices)
            ).item()
            if correct_order is True:
                if verbose is True:
                    print("Correct order identified")
                return (
                    True,
                    j_hamming(true_jac, matrix),
                    j_acc(true_jac, matrix),
                    permutation_loss(s_dag.doubly_stochastic_matrix).item(),
                    permutation_loss(s_ica.doubly_stochastic_matrix).item(),
                )

        optim.step()

    learned_order = s_dag.doubly_stochastic_matrix.max(1)[1]
    correct_order = torch.tensor(permute_indices)

    indices = torch.arange(dim)

    # calculate the ratio of index pairs that are in the correct order
    correct_rank_pairs = 0.0
    if len(learned_order.unique()) == dim:

        for o1 in range(dim):
            for o2 in range(o1 + 1, dim):
                correct_rank_pairs += (
                    indices[learned_order == o1] - indices[learned_order == o2]
                ).sign() == (
                    indices[correct_order == o1] - indices[correct_order == o2]
                ).sign()
    else:
        correct_rank_pairs = -1

    if verbose is True:
        print("----------------------------------")
        print(f"{true_jac=}")
        print(f"{est_jac=}")
        if dag_permute is True:
            matrix = (
                s_ica.doubly_stochastic_matrix
                @ est_jac
                @ s_dag.doubly_stochastic_matrix
            ).detach()
        else:
            matrix = (s_ica.doubly_stochastic_matrix @ est_jac).detach()
        print(matrix)
        if dag_permute is True:
            print(f"S_DAG={s_dag.doubly_stochastic_matrix.detach()}")
        print(f"S_ICA={s_ica.doubly_stochastic_matrix.detach()}")
    return (
        False if rank_acc is False else correct_rank_pairs / (dim * (dim - 1) / 2.0),
        j_hamming(true_jac, matrix),
        j_acc(true_jac, matrix),
        permutation_loss(s_dag.doubly_stochastic_matrix).item(),
        permutation_loss(s_ica.doubly_stochastic_matrix).item(),
    )
