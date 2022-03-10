import torch

from care_nl_ica.models.mlp import ARMLP


def test_assembled_weight_setter():
    num_vars = 3
    armlp = ARMLP(num_vars)

    armlp.assembled_weight = torch.tril(torch.randn_like(armlp.assembled_weight))

    assert armlp.assembled_weight.requires_grad is True
