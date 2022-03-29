import itertools

import torch


def indirect_causes(gt_jacobian_encoder) -> torch.Tensor:
    """
    Calculates all indirect paths in the encoder (SEM/SCM)
    :param gt_jacobian_encoder:
    :return:
    """

    # calculate the indirect cause mask
    eps = 1e-6
    direct_causes = torch.tril((gt_jacobian_encoder.abs() > eps).float(), -1)

    # add together the matrix powers of the adjacency matrix
    # this yields all indirect paths
    paths = graph_paths(direct_causes)

    indirect_causes = torch.stack(list(paths.values())).sum(0)

    indirect_causes = (
        indirect_causes.bool().float()
    )  # convert all non-1 value to 1 (for safety)
    # correct for causes where both the direct and indirect paths are present
    indirect_causes = indirect_causes * ((indirect_causes - direct_causes) > 0).float()

    return indirect_causes, paths


def graph_paths(direct_causes: torch.Tensor) -> dict:
    paths = dict()
    matrix_power = direct_causes.clone()

    for i in range(direct_causes.shape[0]):
        if matrix_power.sum() == 0:
            break

        paths[i] = matrix_power
        matrix_power = matrix_power @ direct_causes

    return paths


def false_positive_paths(
    dep_mat, gt_paths: dict, threshold: float = 1e-2, weighted: bool = False
) -> torch.Tensor:
    direct_causes = torch.tril((dep_mat.abs() > threshold).float(), -1)
    dep_mat_paths = graph_paths(direct_causes)

    weighting = lambda gt_path, path: (
        (1 - gt_path)
        * path
        * (dep_mat if weighted is True else torch.ones_like(dep_mat))
    ).sum()

    return torch.Tensor(
        [
            weighting(gt_path, path)
            for gt_path, path in zip(gt_paths.values(), dep_mat_paths.values())
        ]
    )


def false_negative_paths(
    dep_mat, gt_paths: dict, threshold: float = 1e-2, weighted: bool = False
) -> torch.Tensor:
    direct_causes = torch.tril((dep_mat.abs() > threshold).float(), -1)
    dep_mat_paths = graph_paths(direct_causes)

    weighting = lambda gt_path, path: (
        (1 - path[gt_path.bool()])
        * (dep_mat if weighted is True else torch.ones_like(dep_mat))
    ).sum()

    return torch.Tensor(
        [
            weighting(gt_path, path)
            for gt_path, path in zip(gt_paths.values(), dep_mat_paths.values())
        ]
    )


def causal_orderings(gt_jacobian_encoder) -> list:
    """
    The function calculates the possible causal orderings based on the adjacency matrix

    :param gt_jacobian_encoder:
    :return:
    """
    dim = gt_jacobian_encoder.shape[0]

    # get all indices for nonzero elements
    nonzero_indices = (gt_jacobian_encoder.abs() > 0).nonzero()

    smallest_idx = []
    biggest_idx = []
    idx_range = []

    for i in range(dim):
        # select nonzero indices for the current row
        # and take the column index of the first element
        # this gives the smallest index of variable "i" in the causal ordering
        nonzero_in_row = nonzero_indices[nonzero_indices[:, 0] == i, :]

        # 1. when the first non-zero element is on the main diagonal, then the variable has no parents
        # 2. when the 0th element is nonzero, then i is the smallest index
        # 3. otherwise, pick the first parent's smallest index and add 1
        if (tmp := nonzero_in_row[0][1]) == i:
            smallest_idx.append(0)
        elif tmp == 0:
            smallest_idx.append(i)
        else:
            smallest_idx.append(smallest_idx[tmp] + 1)

        # select nonzero indices for the current columns
        # and take the row index of the first element
        # this gives the biggest index of variable "i" in the causal ordering
        nonzero_in_col = nonzero_indices[nonzero_indices[:, 1] == i, :]

        biggest_idx.append(nonzero_in_col[0][0].item())

        # this means that there is only 1 appearance of variable i,
        # so it can be everywhere in the causal ordering
        if (
            len(nonzero_in_row) == 1
            and len(nonzero_in_col) == 1
            and smallest_idx[i] == i
            and biggest_idx[i] == i
        ):

            idx_range.append(list(range(dim)))
        else:

            idx_range.append(list(range(smallest_idx[i], biggest_idx[i] + 1)))

    orderings = [x for x in itertools.product(*idx_range) if len(set(x)) == dim]

    return orderings
