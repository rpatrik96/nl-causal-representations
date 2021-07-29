import torch

from src.hsic import HSIC


class IndependenceChecker(object):
    """
    Class for encapsulating independence test-related methods
    """

    def __init__(self, hparams) -> None:
        super().__init__()
        self.hparams = hparams

        self.test = HSIC(hparams.num_permutations)

    def check_bivariate_dependence(self, x1, x2):
        decisions = []
        var_map = [1, 1, 2, 2]
        with torch.no_grad():
            decisions.append(self.test.run_test(x1[:, 0], x2[:, 1], device="cpu", bonferroni=4).item())
            decisions.append(self.test.run_test(x1[:, 0], x2[:, 0], device="cpu", bonferroni=4).item())
            decisions.append(self.test.run_test(x1[:, 1], x2[:, 0], device="cpu", bonferroni=4).item())
            decisions.append(self.test.run_test(x1[:, 1], x2[:, 1], device="cpu", bonferroni=4).item())

        return decisions, var_map

    def check_multivariate_dependence(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Carries out HSIC for the multivariate case, all pairs are tested
        :param x1: tensor of the first batch of variables in the shape of (num_elem, num_dim)
        :param x2: tensor of the second batch of variables in the shape of (num_elem, num_dim)
        :return: the adjacency matrix
        """
        num_dim = x1.shape[-1]
        max_edge_num = num_dim ** 2
        adjacency_matrix = torch.zeros(num_dim, num_dim).bool()

        with torch.no_grad():
            for i in range(num_dim):
                for j in range(num_dim):
                    adjacency_matrix[i, j] = self.test.run_test(x1[:, i], x2[:, j], device="cpu",
                                                                bonferroni=max_edge_num).item()

        return adjacency_matrix
