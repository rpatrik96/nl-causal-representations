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
):

    est_jac = torch.from_numpy(est_jac).float()
    true_jac = torch.from_numpy(true_jac).float()
    # print(est_jac)
    dim = est_jac.shape[0]

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
    if True or dag_permute is True:
        for i in range(num_steps):
            # print(f"S_ICA={s_ica.doubly_stochastic_matrix.detach()}")

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
            loss_diag = diag_weight * (1.0 / (matrix.diag() + 1e-8)).sum()
            # loss_diag = -diag_weight * (matrix.diag()).sum()

            loss = loss_l + loss_u + loss_diag
            # print(f"{loss=}")

            loss.backward()

            if i % 250 == 0 and dag_permute is True:
                correct_order = torch.all(
                    s_dag.doubly_stochastic_matrix.max(1)[1]
                    == torch.tensor(permute_indices)
                ).item()
                if correct_order is True:
                    if verbose is True:
                        # print(true_jac)
                        # print(matrix)
                        # print(true_jac.reshape(-1,), matrix.detach().abs().reshape(-1,) > hamming_threshold)
                        print("Correct order identified")
                    return (
                        True,
                        hamming(
                            true_jac.abs().reshape(
                                -1,
                            )
                            > hamming_threshold,
                            matrix.detach()
                            .abs()
                            .reshape(
                                -1,
                            )
                            > hamming_threshold,
                        ),
                        (
                            (
                                true_jac.abs().reshape(
                                    -1,
                                )
                                > 1e-8
                            )
                            == (
                                matrix.detach()
                                .abs()
                                .reshape(
                                    -1,
                                )
                                > hamming_threshold
                            )
                        )
                        .float()
                        .mean(),
                    )

            optim.step()
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
            # matrix = est_jac[torch.argsort(est_jac.sum(1).long()-1), :]
        print(matrix)
        if dag_permute is True:
            print(f"S_DAG={s_dag.doubly_stochastic_matrix.detach()}")
        print(f"S_ICA={s_ica.doubly_stochastic_matrix.detach()}")
    return (
        False,
        hamming(
            true_jac.abs().reshape(
                -1,
            )
            > 1e-8,
            matrix.detach()
            .abs()
            .reshape(
                -1,
            )
            > hamming_threshold,
        ),
        (
            (
                true_jac.abs().reshape(
                    -1,
                )
                > hamming_threshold
            )
            == (
                matrix.detach()
                .abs()
                .reshape(
                    -1,
                )
                > hamming_threshold
            )
        )
        .float()
        .mean(),
    )


if __name__ == "__main__":

    from care_nl_ica.losses.dep_mat import permutation_loss

    NUM_DIM = 3
    s = SinkhornNet(NUM_DIM, 20, 1e-4)
    s2 = SinkhornNet(NUM_DIM, 20, 1e-4)
    optim = torch.optim.Adam(list(s.parameters()) + list(s2.parameters()), lr=1e-3)
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
    ).abs()
    dim = J_permuted.shape[0]
    # zero_idx = J_permuted.abs().view(-1,).sort()[1][:dim*(dim-1)//2]
    # J_permuted.view(-1,)[zero_idx]=0
    #
    # smallest_item = J_permuted[J_permuted.abs()>0].abs().view(-1, ).sort()[0][0]
    # J_permuted[J_permuted.abs() == smallest_item] = 0
    #
    # zeros_in_rows = (J_permuted == 0).sum(1)
    # if zeros_in_rows.max() >= dim:
    #     raise ValueError

    #
    J_permuted = torch.tensor(
        [
            [0.00099, -1.8380, 0.0052],
            [0.00181899, 1.3469636440, -1.227944],
            [1.874460, 0.9891961, -0.900583],
        ]
    ).abs()
    #
    # J = 1.0 - J_permuted.bool().float() + 1e-5
    #
    # J= J_permuted
    #
    # JJ = J
    # col_order = []
    # for col in range (dim):
    #     if (dim-col) in JJ.sum(0):
    #         col_order.append(torch.arange(dim)[(dim-col) == JJ.sum(0)][0])
    #         JJ = JJ[list(range(dim))]
    # for row in range(dim):
    #     pass

    # for i in range(dim):
    #     if (dim - i) in JJ.sum(0):
    #         col_order.append(JJ.sum(0).tolist().index(dim-i))

    from scipy.optimize import linear_sum_assignment

    print(f"{J_permuted=}")

    for i in range(3000):

        optim.zero_grad()
        matrix = (
            s2.doubly_stochastic_matrix
            @ J_permuted.float()
            @ s.doubly_stochastic_matrix
        )
        loss_u = 10 * torch.triu(matrix, 1).abs().sum()  #
        loss_l = -torch.tril(matrix, 0).abs().sum() * 10
        # loss = loss_l + loss_u

        # loss_order = (torch.tensor([0,1,2]) - (J_permuted@ s.doubly_stochastic_matrix).sum(0)).sum()
        # loss_ica_perm = (torch.tensor([2,1,0]) - (s2.doubly_stochastic_matrix@J_permuted).sum(1)).sum()
        # loss = loss_order #+ loss_ica_perm
        loss = (1.0 / matrix.diag()).sum() + loss_l + loss_u

        # loss += 10*(permutation_loss(s.doubly_stochastic_matrix) +permutation_loss(s2.doubly_stochastic_matrix))

        loss.backward()

        if i % 1000 == 0:
            print(
                f"{loss.item():.3f}, {torch.all(s.doubly_stochastic_matrix.max(1)[1]==torch.tensor(permute_indices))}"
            )
            #     print(s.doubly_stochastic_matrix.detach())
            print(matrix)

        optim.step()

    # print(f"{permute_mat.T=}")
    print(f"S={s.doubly_stochastic_matrix.detach()}")
    print(f"S2={s2.doubly_stochastic_matrix.detach()}")
    print(s2.doubly_stochastic_matrix @ J_permuted.float() @ s.doubly_stochastic_matrix)
    print(
        torch.all(s.doubly_stochastic_matrix.max(1)[1] == torch.tensor(permute_indices))
    )
