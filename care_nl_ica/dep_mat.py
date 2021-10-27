import torch
import torch.nn as nn
from torch.autograd.functional import jacobian


def calc_jacobian(encoder: nn.Module, latents: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Calculate the Jacobian more efficiently than ` torch.autograd.functional.jacobian`
    :param encoder: the model to calculate the Jacobian of
    :param latents: the inputs for evaluating the model
    :param normalize: flag to rescale the Jacobian to have unit norm
    :return: B x n_out x n_in
    """

    jacob = []
    input_vars = latents.clone().requires_grad_(True)

    # set to eval mode but remember original state
    in_training: bool = encoder.training
    encoder.eval()  # otherwise we will get 0 gradients

    output_vars = encoder(input_vars)

    for i in range(output_vars.shape[1]):
        jacob.append(torch.autograd.grad(output_vars[:, i:i + 1], input_vars, create_graph=True,
                                         grad_outputs=torch.ones(output_vars[:, i:i + 1].shape).to(output_vars.device))[
                         0])

    jacobian = torch.stack(jacob, 1)

    # make the Jacobian volume preserving
    if normalize is True:
        jacobian *= jacobian.det().abs().pow(1 / jacobian.shape[0]).reshape(-1, 1, 1)

    # set back to original mode
    if in_training is True:
        encoder.train()

    return jacobian


def calc_jacobian_numerical(model, x, dim, device,  eps=1e-6):
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
        J[:,j] = (model(x+ delta) - model(x)).abs().mean(0) / (2*eps)

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
    This is calculated as the L2 squared norm of the upper triangular part of the dependency matrix
    (except the main diagonal).

    :param dep_mat: dependency matrix as a torch.Tensor
    :return: the triangularity loss as a torch.Tensor (scalar)
    """

    return torch.triu(dep_mat, 1).pow(2).sum()


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


def calc_jacobian_loss(args, f, g, latent_space, device, eps=1e-6, calc_numerical:bool=False):
    # 1. get a sample from the latents
    # these are the noise variables in Wieland's notation
    # todo: probably we can use something from data?
    z_disentanglement = latent_space.sample_marginal(args.n_eval_samples)
    # 2. calculate the signal mixtures (i.e., the observations)
    obs = g(z_disentanglement)
    # 3. calculate the dependency matrix
    # x \times f(x)
    dep_mat = calc_jacobian(f, obs.clone(), normalize=args.preserve_vol).abs().mean(0)
    # 3/b calculate the numerical jacobian
    # calculate numerical jacobian
    numerical_jacobian = None if calc_numerical is False else calc_jacobian_numerical(f, obs,dep_mat.shape[0], device, eps)
    # 4. calculate the loss for the dependency matrix
    dep_loss = dependency_loss(dep_mat)
    return dep_loss, dep_mat, numerical_jacobian
