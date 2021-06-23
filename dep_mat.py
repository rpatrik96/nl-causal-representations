import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

def calc_jacobian(encoder:nn.Module, latents:torch.Tensor)->torch.Tensor:
    #return B x n_out x n_in
    jacob = []
    input_vars = latents.clone().requires_grad_(True)
    output_vars = encoder(input_vars)
    for i in range(output_vars.shape[1]):
        jacob.append(torch.autograd.grad(output_vars[:, i:i+1], input_vars, create_graph=True, 
                                         grad_outputs=torch.ones(output_vars[:,i:i+1].shape).to(output_vars.device))[0])
    return torch.stack(jacob, 1)

def calc_dependency_matrix(encoder:nn.Module, latents:torch.Tensor)->torch.Tensor:
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

def sparsity_loss(dep_mat:torch.Tensor)->torch.Tensor:
    """
    Calculates the sparsity-inducing (i.e., L1) loss for the dependency matrix.

    :param dep_mat: dependency matrix as a torch.Tensor
    :return: the sparsity loss as a torch.Tensor (scalar)
    """
    return dep_mat.abs().sum()

def triangularity_loss(dep_mat:torch.Tensor)->torch.Tensor:
    """
    Calculates the loss term for inducing a **lower** triangular structure for the dependency matrix.
    This is calculated as the L2 squared norm of the upper triangular part of the dependency matrix
    (except the main diagonal).

    :param dep_mat: dependency matrix as a torch.Tensor
    :return: the triangularity loss as a torch.Tensor (scalar)
    """

    return torch.triu(dep_mat, 1).pow(2).sum()

def dependency_loss(dep_mat:torch.Tensor, weight_sparse:float=1., weight_triangular:float=1.)->torch.Tensor:
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


    return weight_sparse*sparse_loss + weight_triangular*triangular_loss



def optimize_dependency_matrix(generator:nn.Module, encoder:nn.Module):

    # todo: dummy function
    # sample latents
    latents = generator.sample()

    # calculate the dependency matrix
    dep_mat = calc_dependency_matrix(encoder, latents)

    dep_loss = dependency_loss(dep_mat)

