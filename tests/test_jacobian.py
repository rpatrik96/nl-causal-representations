import pytest
import torch
from torch.autograd.functional import jacobian

from care_nl_ica.dep_mat import calc_jacobian, calc_jacobian_numerical
from care_nl_ica.models.model import ContrastiveLearningModel
from argparse import Namespace


@pytest.fixture()
def model(args):

    args = Namespace(
        **{
            **args.model,
            **{
                "latent_dim": args.data.latent_dim,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "normalize_latents": args.data.normalize_latents,
            },
        }
    )
    return ContrastiveLearningModel(args)


def test_triangularity_jacobian(
    model: ContrastiveLearningModel,
    dataloader,
    numerical_check: bool = False,
    built_in_jacobian_check: bool = False,
):
    """

    Checks the AR nature of the model by calculating the Jacobian.

    :param model: model to test
    :return:
    """

    # draw a sample from the latent space (marginal only)
    z = next(iter(dataloader))[0, :]

    # calculate the Jacobian
    dep_mat = calc_jacobian(
        model.unmixing, z[0], normalize=model.hparams.normalize_latents
    ).mean(0)
    print(f"{dep_mat=}")

    # numerical Jacobian
    if numerical_check is True:
        print(
            f"{calc_jacobian_numerical(model.unmixing, z, model.hparams.latent_dim, model.hparams.device)=}"
        )

    # same as calc_jacobian, but using the torch jacobian function
    if built_in_jacobian_check is True:
        # x in shape (Batch, Length)
        def _func_sum(x):
            return model.unmixing.forward(x).sum(dim=0)

        print("---------------")

        print(jacobian(_func_sum, z).permute(1, 0, 2).abs().mean(0))

    assert (torch.tril(dep_mat) != dep_mat).sum() == 0


def test_triangularity_naive(model: ContrastiveLearningModel):
    """
    Checks the AR nature of the model by perturbing the input and observing the changes in the outputs.

    :param model: model to test
    :return:
    """

    # constants
    batch_size = 1
    tria_check = torch.zeros(model.hparams.latent_dim)

    # set to eval mode but remember original state
    in_training: bool = model.training
    model.eval()  # otherwise we will get 0 gradients

    # calculate the baseline output - all inputs should be different from 0
    # this is to avoid degenerate cases making the test succeed
    y0 = model.unmixing(
        torch.ones(batch_size, model.hparams.latent_dim).to(model.hparams.device)
    )
    print(f"{y0=}")

    # unsqueeze for the AR MLP
    if len(y0.shape) == 1:
        y0 = y0.unsqueeze(0)

    # loop for perturbing each input one-by-one
    for i in range(model.hparams.latent_dim):
        z = torch.ones(batch_size, model.hparams.latent_dim).to(model.hparams.device)
        z[:, i] = -1

        y = model.unmixing(z)

        print(f"{i=},{y=}")

        # unsqueeze for the AR MLP
        if len(y.shape) == 1:
            y = y.unsqueeze(0)

        # the indexing is up to the ith element
        # as input i affects outputs i:n
        # so a change before that is a failure
        tria_check[i] = (y[:, :i] != y0[:, :i]).sum()

    # set back to original mode
    if in_training is True:
        model.train()

    assert tria_check.sum() == 0
