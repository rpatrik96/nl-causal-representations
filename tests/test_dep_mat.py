from argparse import Namespace

import torch

from care_nl_ica.dep_mat import calc_jacobian
from care_nl_ica.models.model import ContrastiveLearningModel


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
