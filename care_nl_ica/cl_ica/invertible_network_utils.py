"""Create invertible mixing networks."""

import numpy as np
import torch
from torch import nn
from scipy.stats import ortho_group
from typing import Union
from typing_extensions import Literal
import care_nl_ica.cl_ica.encoders

from typing import Tuple

__all__ = ["construct_invertible_flow", "construct_invertible_mlp"]


def construct_invertible_mlp(
    n: int = 20,
    n_layers: int = 2,
    n_iter_cond_thresh: int = 10000,
    cond_thresh_ratio: float = 0.25,
    weight_matrix_init: Union[Literal["pcl"], Literal["rvs"]] = "pcl",
    act_fct: Union[
        Literal["relu"],
        Literal["leaky_relu"],
        Literal["elu"],
        Literal["smooth_leaky_relu"],
        Literal["softplus"],
    ] = "leaky_relu",
    lower_triangular=True,
    sparsity=True,
    variant=None,
    offset=0,
):
    """
    Create an (approximately) invertible mixing network based on an MLP.
    Based on the mixing code by Hyvarinen et al.

    Args:
        n: Dimensionality of the input and output data
        n_layers: Number of layers in the MLP.
        n_iter_cond_thresh: How many random matrices to use as a pool to find weights.
        cond_thresh_ratio: Relative threshold how much the invertibility
            (based on the condition number) can be violated in each layer.
        weight_matrix_init: How to initialize the weight matrices.
        act_fct: Activation function for hidden layers.
        :param offset:
    """

    class SmoothLeakyReLU(nn.Module):
        def __init__(self, alpha=0.2):
            super().__init__()
            self.alpha = alpha

        def forward(self, x):
            return self.alpha * x + (1 - self.alpha) * torch.log(1 + torch.exp(x))

    def get_act_fct(act_fct):
        if act_fct == "relu":
            return torch.nn.ReLU, {}, 1
        if act_fct == "leaky_relu":
            return torch.nn.LeakyReLU, {"negative_slope": 0.2}, 1
        elif act_fct == "elu":
            return torch.nn.ELU, {"alpha": 1.0}, 1
        elif act_fct == "max_out":
            raise NotImplemented()
        elif act_fct == "smooth_leaky_relu":
            return SmoothLeakyReLU, {"alpha": 0.2}, 1
        elif act_fct == "softplus":
            return torch.nn.Softplus, {"beta": 1}, 1
        else:
            raise Exception(f"activation function {act_fct} not defined.")

    layers = []
    act_fct, act_kwargs, act_fac = get_act_fct(act_fct)

    # Subfuction to normalize mixing matrix
    def l2_normalize(Amat, axis=0):
        # axis: 0=column-normalization, 1=row-normalization
        l2norm = np.sqrt(np.sum(Amat * Amat, axis))
        Amat = Amat / l2norm
        return Amat

    condList = np.zeros([n_iter_cond_thresh])
    if weight_matrix_init == "pcl":
        for i in range(n_iter_cond_thresh):
            A = np.random.uniform(-1, 1, [n, n])
            A = l2_normalize(A, axis=0)
            condList[i] = np.linalg.cond(A)
        condList.sort()  # Ascending order
    condThresh = condList[int(n_iter_cond_thresh * cond_thresh_ratio)]
    print("condition number threshold: {0:f}".format(condThresh))

    for i in range(n_layers):

        lin_layer = nn.Linear(n, n, bias=False)
        if weight_matrix_init == "pcl" or weight_matrix_init == "rvs":
            if weight_matrix_init == "pcl":
                condA = condThresh + 1
                while condA > condThresh:
                    weight_matrix = np.random.uniform(-1, 1, (n, n))
                    weight_matrix = l2_normalize(weight_matrix, axis=0)

                    condA = np.linalg.cond(weight_matrix)
                    # print("    L{0:d}: cond={1:f}".format(i, condA))
                print(
                    f"layer {i+1}/{n_layers},  condition number: {np.linalg.cond(weight_matrix)}"
                )
            elif weight_matrix_init == "rvs":
                weight_matrix = ortho_group.rvs(n)

                if offset != 0.0:
                    scaling = offset * np.eye(n)
                    weight_matrix = scaling @ weight_matrix
            if lower_triangular:
                if sparsity:
                    _, tril_mask = createARmask(n, variant)
                    weight_matrix = tril_mask * weight_matrix
                else:
                    weight_matrix = np.tril(weight_matrix)
            lin_layer.weight.data = torch.tensor(weight_matrix, dtype=torch.float32)
        elif weight_matrix_init == "expand":
            pass
        else:
            raise Exception(f"weight matrix {weight_matrix_init} not implemented")

        layers.append(lin_layer)

        if i < n_layers - 1:
            layers.append(act_fct(**act_kwargs))

    mixing_net = nn.Sequential(*layers)

    # fix parameters
    for p in mixing_net.parameters():
        p.requires_grad = False

    return mixing_net


def tensor2bitlist(x: torch.IntTensor, bits: int) -> torch.Tensor:
    """
    Converts an integer into a list of binary tensors.

    Source https://stackoverflow.com/a/63546308

    :param x: number to convert into binary
    :param bits: number of bits to use
    :return:
    """

    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def createARmask(
    dim: int, variant: torch.IntTensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a (sparse) autoregressive triangular mask

    :param dim: dimensionality of the matrix
    :return:
    """

    # constants
    mask_numel = dim * (dim - 1) // 2
    row_idx, col_idx = torch.tril_indices(dim, dim, -1)

    if variant is None:
        max_variants = 2**mask_numel
        variant = torch.randint(max_variants, (1,)).int()

    # create mask elements
    mask_elem = tensor2bitlist(variant, mask_numel)

    # fill the mask
    mask = torch.eye(dim)
    mask[row_idx, col_idx] = mask_elem.float()

    return variant, mask


def construct_invertible_flow(
    n: int,
    coupling_block: Union[Literal["gin", "glow"]] = "gin",
    num_nodes: int = 8,
    node_size_factor: int = 1,
):
    """
    Creates an invertible mixing based through a flow-based network.

    Args:
        n: Dimensionality of the input and output data
        coupling_block: Coupling method to use to combine nodes.
        num_nodes: Depth of the flow network.
        node_size_factor: Hidden units per node.
    """

    return encoders.get_flow(n, n, False, coupling_block, num_nodes, node_size_factor)
