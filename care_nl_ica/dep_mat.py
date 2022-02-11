import itertools

import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

from care_nl_ica.metrics.metrics import JacobianMetrics, amari_distance, permutation_loss, extract_permutation_from_jacobian


def calc_jacobian(model: nn.Module, latents: torch.Tensor, normalize: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """
    Calculate the Jacobian more efficiently than ` torch.autograd.functional.jacobian`
    :param model: the model to calculate the Jacobian of
    :param latents: the inputs for evaluating the model
    :param normalize: flag to rescale the Jacobian to have unit norm
    :param eps: the epsilon to use for numerical stability
    :return: B x n_out x n_in
    """

    jacob = []
    input_vars = latents.clone().requires_grad_(True)

    # set to eval mode but remember original state
    in_training: bool = model.training
    model.eval()  # otherwise we will get 0 gradients

    output_vars = model(input_vars)

    for i in range(output_vars.shape[1]):
        jacob.append(torch.autograd.grad(output_vars[:, i:i + 1], input_vars, create_graph=True,
                                         grad_outputs=torch.ones(output_vars[:, i:i + 1].shape).to(output_vars.device))[
                         0])

    jacobian = torch.stack(jacob, 1)

    if normalize is True:
        # normalize the Jacobian by making it volume preserving
        # jacobian /= jacobian.det().abs().pow(1 / jacobian.shape[-1]).reshape(-1, 1, 1)

        # normalize to make variance to 1
        # norm_factor = (output_vars.std(dim=0) + 1e-8)
        # jacobian /= norm_factor.reshape(1, 1, -1)

        # normalize range to [0;1]
        dim_range = (output_vars.max(dim=0)[0] - output_vars.min(dim=0)[0]).abs()

        jacobian /= (dim_range + eps)

    # set back to original mode
    if in_training is True:
        model.train()

    return jacobian


def calc_jacobian_numerical(model, x, dim, device, eps=1e-6):
    """
    Calculate the Jacobian numerically
    :param model: the model to calculate the Jacobian of
    :param x: the inputs for evaluating the model with dimensions B x n_in
    :param dim: the dimensionality of the output
    :param device: the device to calculate the Jacobian on
    :param eps: the epsilon to use for numerical differentiation

    :return: n_out x n_in
    """

    # set to eval mode but remember original state
    in_training: bool = model.training
    model.eval()  # otherwise we will get 0 gradients

    # clone input to avoid problems
    x = x.clone().requires_grad_(True)

    # init jacobian
    J = torch.zeros(dim, x.shape[1])

    # iterate over input dims and perturb
    for j in range(dim):
        delta = torch.zeros(dim).to(device)
        delta[j] = eps
        J[:, j] = (model(x + delta) - model(x)).abs().mean(0) / (2 * eps)

    # reset to original state
    if in_training is True:
        model.train()

    return J


def calc_dependency_matrix(encoder: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """
    Calculates the dependecy matrix, which is
    :param encoder: a mapping f from the latent factors Z to the noise variables N
    :param latents: tensor of the latent variables Z
    :return: the dependency matrix, i.e. |J_f(Z)|
    """

    # calculate the jacobian
    jacob = jacobian(encoder.forward, latents)
    # take the absolute value
    return jacob.abs()


def sparsity_loss(dep_mat: torch.Tensor) -> torch.Tensor:
    """
    Calculates the sparsity-inducing (i.e., L1) loss for the dependency matrix.

    :param dep_mat: dependency matrix as a torch.Tensor
    :return: the sparsity loss as a torch.Tensor (scalar)
    """
    return dep_mat.abs().sum()


def triangularity_loss(dep_mat: torch.Tensor) -> torch.Tensor:
    """
    Calculates the loss term for inducing a **lower** triangular structure for the dependency matrix.
    This is calculated as the L1 norm of the upper triangular part of the dependency matrix
    (except the main diagonal).

    :param dep_mat: dependency matrix as a torch.Tensor
    :return: the triangularity loss as a torch.Tensor (scalar)
    """

    return torch.triu(dep_mat, 1).abs().mean()


def dependency_loss(dep_mat: torch.Tensor, weight_sparse: float = 1., weight_triangular: float = 1.) -> torch.Tensor:
    """
    Calculates the weighted sum of the triangularity-enforcing and the sparsity-inducing losses for the
    dependency matrix.

    :param dep_mat: dependency matrix as a torch.Tensor
    :param weight_sparse: scalar for weighting the sparsity loss
    :param weight_triangular: scalar for weighting the triangularity loss
    :return: sum of the two losses as a torch.Tensor
    """

    sparse_loss = sparsity_loss(dep_mat)
    triangular_loss = triangularity_loss(dep_mat)

    return weight_sparse * sparse_loss + weight_triangular * triangular_loss


def calc_jacobian_loss(model, latent_space, eps=1e-6, calc_numerical: bool = False):
    args = model.hparams
    # 1. get a sample from the latents
    # these are the noise variables in Wieland's notation
    # todo: probably we can use something from data?
    z_disentanglement = latent_space.sample_marginal(args.n_eval_samples)
    # 2. calculate the signal mixtures (i.e., the observations)
    obs = model.decoder(z_disentanglement.clone())
    # 3. calculate the dependency matrix
    # x \times f(x)
    dep_mat = calc_jacobian(model.encoder, obs.clone(), normalize=args.normalize_latents).mean(0)

    jac_enc_dec = calc_jacobian(model, z_disentanglement.clone(), normalize=args.normalize_latents).mean(0)

    # 3/b calculate the numerical jacobian
    # calculate numerical jacobian
    numerical_jacobian = None if calc_numerical is False else calc_jacobian_numerical(model.encoder, obs,
                                                                                      dep_mat.shape[0], args.device,
                                                                                      eps)
    # 4. calculate the loss for the dependency matrix
    dep_loss = dependency_loss(dep_mat)
    return triangularity_loss(dep_mat), dep_mat, numerical_jacobian, jac_enc_dec


def dep_mat_metrics(dep_mat: torch.Tensor, gt_jacobian_encoder, indirect_causes,
                    gt_jacobian_decoder_permuted, threshold: float = 1e-3) -> JacobianMetrics:
    # calculate the optimal threshold for 1 accuracy
    # calculate the indices where the GT is 0 (in the lower triangular part)
    sparsity_mask = (torch.tril(gt_jacobian_encoder.abs() < 1e-6)).bool()

    if sparsity_mask.sum() > 0:
        optimal_threshold = dep_mat[sparsity_mask].abs().max()
    else:
        optimal_threshold = None

    # calculate the distance between ground truth and predicted jacobian
    norm_diff: float = torch.norm(dep_mat.abs() - gt_jacobian_encoder.abs()).mean()
    thresholded_norm_diff: float = torch.norm(
        dep_mat.abs() * (dep_mat.abs() > threshold) - gt_jacobian_encoder.abs()).mean()

    # calculate the fraction of correctly identified zeroes
    incorrect_edges: float = ((dep_mat.abs() * indirect_causes) > threshold).sum()
    sparsity_accuracy: float = 1. - incorrect_edges / (indirect_causes.sum() + 1e-8)

    metrics = JacobianMetrics(norm_diff, thresholded_norm_diff, optimal_threshold, sparsity_accuracy,
                              amari_distance(dep_mat, gt_jacobian_decoder_permuted),
                              permutation_loss(extract_permutation_from_jacobian(dep_mat, qr=True), matrix_power=True))

    return metrics


def calc_indirect_causes(gt_jacobian_encoder) -> torch.Tensor:
    """
    Calculates all indirect paths in the encoder (SEM/SCM)
    :param gt_jacobian_encoder:
    :return:
    """

    # calculate the indirect cause mask
    eps = 1e-6
    matrix_power = direct_causes = torch.tril((gt_jacobian_encoder.abs() > eps).float(), -1)
    indirect_causes = torch.zeros_like(gt_jacobian_encoder)

    # add together the matrix powers of the adjacency matrix
    # this yields all indirect paths
    for i in range(gt_jacobian_encoder.shape[0]):
        matrix_power = matrix_power @ direct_causes
        if matrix_power.sum() == 0:
            break

        indirect_causes += matrix_power

    indirect_causes = indirect_causes.bool().float()  # convert all non-1 value to 1 (for safety)
    # correct for causes where both the direct and indirect paths are present
    indirect_causes = indirect_causes * ((indirect_causes - direct_causes) > 0).float()

    return indirect_causes


def calc_causal_orderings(gt_jacobian_encoder) -> list:
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

        smallest_idx.append(0 if (tmp := nonzero_in_row[0][1]) == i else smallest_idx[tmp] + 1)

        # select nonzero indices for the current columns
        # and take the row index of the first element
        # this gives the biggest index of variable "i" in the causal ordering
        nonzero_in_col = nonzero_indices[nonzero_indices[:, 1] == i, :]

        biggest_idx.append(nonzero_in_col[0][0])

        # this means that there is only 1 appearance of variable i,
        # so it can be everywhere in the causal ordering
        if len(nonzero_in_row) == 1 and len(nonzero_in_col) == 1 and smallest_idx[i] == i and biggest_idx[i] == i:

            idx_range.append(list(range(dim)))
        else:

            idx_range.append(list(range(smallest_idx[i], biggest_idx[i] + 1)))

    orderings = [x for x in list(itertools.product(*idx_range)) if len(set(x)) == dim]

    return orderings
