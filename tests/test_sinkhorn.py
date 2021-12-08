from care_nl_ica.sinkhorn import SinkhornNet
import torch

def test_sinkhorn_net():
    num_vars = 3
    num_steps = 30
    temperature = 7e-3
    threshold = .95

    sinkhorn_net = SinkhornNet(num_vars, num_steps, temperature)

    assert torch.all((sinkhorn_net.doubly_stochastic_matrix.abs()>threshold).sum(dim=0) == torch.ones(1,num_vars))