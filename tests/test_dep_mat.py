import torch

from care_nl_ica.metrics.dep_mat import check_permutation


def test_check_permutation():
    num_dim = 5
    threshold = 0.95
    eps = 1e-3
    soft_perm = eps * torch.ones(num_dim, num_dim)
    soft_perm[torch.randperm(5), torch.randperm(5)] = threshold + eps * num_dim

    soft_perm /= soft_perm.sum(0, keepdim=True)
    soft_perm /= soft_perm.sum(1, keepdim=True)

    hard_permutation, success = check_permutation(soft_perm, threshold)

    assert success is True
