from sinkhorn import SinkhornNet
import torch
from care_nl_ica.utils import setup_seed

def test_sinkhorn_net():
    num_vars = 3
    num_steps = 5
    temperature = 3e-2
    threshold = .95

    setup_seed(1)

    sinkhorn_net = SinkhornNet(num_vars, num_steps, temperature)

    print(sinkhorn_net.doubly_stochastic_matrix.abs())
    assert torch.all((sinkhorn_net.doubly_stochastic_matrix.abs()>threshold).sum(dim=0) == torch.ones(1,num_vars))