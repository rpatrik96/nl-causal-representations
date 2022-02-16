import pytest
import torch

from care_nl_ica.graph_utils import indirect_causes, graph_paths, false_positive_paths, false_negative_paths


@pytest.fixture
def three_dim_chain()->torch.Tensor:
    dim = 3
    dep_mat = torch.tril(torch.ones(dim, dim))
    zeros_in_chain = torch.tril(torch.ones_like(dep_mat), -2)
    dep_mat[zeros_in_chain == 1] = 0

    return dep_mat


def test_graph_paths(three_dim_chain:torch.Tensor):
    gt_paths = {0: torch.Tensor([[0., 0., 0.],
                              [1., 0., 0.],
                              [0., 1., 0.]]),
             1: torch.Tensor([[0., 0., 0.],
                              [1., 0., 0.],
                              [0., 1., 0.]])}

    [torch.all(path, gt_path) for path, gt_path in zip(graph_paths(three_dim_chain).values(), gt_paths)]


def test_indirect_causes(three_dim_chain:torch.Tensor):
    # ground truth
    indirect_paths = torch.zeros_like(three_dim_chain)
    indirect_paths[-1, 0] = 1

    print(indirect_causes(three_dim_chain)[1])

    torch.all(indirect_causes(three_dim_chain)[0] == indirect_paths)

@pytest.mark.parametrize("weighted", [True, False])
def test_false_positive_paths(three_dim_chain:torch.Tensor, weighted):
    # false positive at [2,0]
    dep_mat = torch.tril(torch.rand_like(three_dim_chain))+1.3
    direct_causes = torch.tril((three_dim_chain.abs() > 1e-6).float(), -1)

    fp = torch.Tensor([1 if weighted is False else dep_mat[2,0], 0])
    torch.all(false_positive_paths(dep_mat, graph_paths(direct_causes), weighted=weighted) == fp)


@pytest.mark.parametrize("weighted", [True, False])
def test_false_negative_paths(three_dim_chain:torch.Tensor, weighted):
    gt_adjacency = torch.tril(torch.rand_like(three_dim_chain))+1.3
    direct_causes = torch.tril((gt_adjacency.abs() > 1e-6).float(), -1)


    # false negative at [2,0] - here three_dim_chain is the estimated value
    fn = torch.Tensor([1 if weighted is False else gt_adjacency[2,0], 0])
    torch.all(false_negative_paths(three_dim_chain, graph_paths(direct_causes), weighted=weighted) == fn)
