import torch
import torch.nn as nn

from torch.autograd.functional import jacobian


def calc_jacobian(
    model: nn.Module,
    latents: torch.Tensor,
    normalize: bool = False,
    eps: float = 1e-8,
    vectorize=False,
    reverse_ad=True,
    norm_range=True,
    norm_diagonal=False,
    output_idx=None,
    aux_inputs=None,
) -> torch.Tensor:
    """
    Calculate the Jacobian more efficiently than ` torch.autograd.functional.jacobian`
    :param aux_inputs:
    :param output_idx: if model output is a tuple, choose this idx
    :param norm_range:
    :param norm_diagonal:
    :param reverse_ad: use reverse mode auto-differentiation (e.g., PReLU only supports this)
    :param vectorize: use functorch vectorization
    :param model: the model to calculate the Jacobian of
    :param latents: the inputs for evaluating the model
    :param normalize: flag to rescale the Jacobian to have unit norm
    :param eps: the epsilon to use for numerical stability
    :return: B x n_out x n_in
    """
    # set to eval mode but remember original state
    in_training: bool = model.training
    model.eval()  # otherwise we will get 0 gradients
    with torch.set_grad_enabled(True):
        jacob = []
        input_vars = latents.clone().requires_grad_(True)

        if aux_inputs is None:
            output_vars = model(input_vars)
        else:
            output_vars = model(input_vars, aux_inputs)

        if output_idx is not None:
            output_vars = output_vars[output_idx]

        if not vectorize:

            for i in range(output_vars.shape[1]):
                jacob.append(
                    torch.autograd.grad(
                        output_vars[:, i : i + 1],
                        input_vars,
                        create_graph=True,
                        grad_outputs=torch.ones(output_vars[:, i : i + 1].shape).to(
                            output_vars.device
                        ),
                    )[0].detach()
                )

            jacobian = torch.stack(jacob, 1)
        else:
            from functorch import vmap, jacrev, jacfwd

            if reverse_ad is True:
                jac_fn = jacrev
            else:
                jac_fn = jacfwd

            sample_jacobian = jac_fn(model.forward, argnums=0)
            jacobian = vmap(
                lambda x: sample_jacobian(torch.unsqueeze(x, 0)), in_dims=0
            )(input_vars).squeeze()

    if normalize is True:
        # normalize the Jacobian by making it volume preserving
        # jacobian /= jacobian.det().abs().pow(1 / jacobian.shape[-1]).reshape(-1, 1, 1)

        # normalize to make variance to 1
        # norm_factor = (output_vars.std(dim=0) + 1e-8)
        # jacobian /= norm_factor.reshape(1, 1, -1)
        if norm_range is True:
            # normalize range to [0;1]
            dim_range = (
                (output_vars.max(dim=0)[0] - output_vars.min(dim=0)[0])
                .abs()
                .reshape(-1, 1)
            )

            jacobian /= dim_range + eps
        elif norm_diagonal is True:
            assert (dim := jacobian.shape[1]) == jacobian.shape[2]
            jacobian /= jacobian[:, (r := torch.arange(dim)), r].unsqueeze(-1) + eps

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


def jacobians(unmixing, sources, mixtures, eps=1e-6, calc_numerical: bool = False):
    # calculate the dependency matrix
    dep_mat = (
        calc_jacobian(
            unmixing, mixtures.clone(), normalize=unmixing.hparams.normalize_latents
        )
        .abs()
        .mean(0)
    )

    jac_enc_dec = (
        calc_jacobian(
            unmixing, sources.clone(), normalize=unmixing.hparams.normalize_latents
        )
        .abs()
        .mean(0)
    )

    # 3/b calculate the numerical jacobian
    # calculate numerical jacobian
    numerical_jacobian = (
        None
        if calc_numerical is False
        else calc_jacobian_numerical(
            unmixing, mixtures, dep_mat.shape[0], unmixing.hparams.device, eps
        )
    )

    return dep_mat, numerical_jacobian, jac_enc_dec
