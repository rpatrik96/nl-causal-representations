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
