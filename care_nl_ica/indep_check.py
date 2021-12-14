from collections import Counter

import torch

from care_nl_ica.dep_mat import calc_jacobian
from care_nl_ica.hsic import HSIC
from care_nl_ica.prob_utils import calc_disentanglement_scores


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

    def check_independence_z_gz(self, decoder, latent_space)->torch.Tensor:
        z_disentanglement = latent_space.sample_marginal(self.hparams.n_eval_samples)
        lin_dis_score, perm_dis_score = calc_disentanglement_scores(z_disentanglement, decoder(z_disentanglement))

        print(f"Id. Lin. Disentanglement: {lin_dis_score:.4f}")
        print(f"Id. Perm. Disentanglement: {perm_dis_score:.4f}")
        print('Run test with ground truth sources')

        dep_mat = None
        if self.hparams.use_dep_mat:
            # x \times z
            dep_mat = calc_jacobian(decoder, z_disentanglement, normalize=self.hparams.normalize_latents).abs().mean(0)

            if self.hparams.permute is True:
                dep_mat = decoder.permutation_matrix.T@dep_mat@decoder.permutation_matrix

            print(dep_mat)
            null_list = [False] * torch.numel(dep_mat)
            null_list[torch.argmin(dep_mat).item()] = True
            var_map = [1, 1, 2, 2]
        else:
            null_list, var_map = self.check_bivariate_dependence(decoder(z_disentanglement), z_disentanglement)
        ######Note this is specific to a dense 2x2 triangular matrix!######
        if Counter(null_list) == Counter([False, False, False, True]):

            print('concluded a causal effect')

            for i, null in enumerate(null_list):
                if null:
                    print('cause variable is X{}'.format(str(var_map[i])))

        else:
            print('no causal effect...?')
            # sys.exit()

        return dep_mat
