from argparse import Namespace

import torch

from care_nl_ica.dep_mat import calc_jacobian
from care_nl_ica.metrics.dep_mat import check_permutation
from care_nl_ica.models.model import ContrastiveLearningModel


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


def test_calc_jacobian(args):
    m = ContrastiveLearningModel(
        Namespace(
            **{
                **args.model,
                **args.data,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
        )
    )
    x = torch.randn(64, args.data.latent_dim)

    assert (
        torch.allclose(
            calc_jacobian(m, x, vectorize=True), calc_jacobian(m, x, vectorize=False)
        )
        == True
    )
