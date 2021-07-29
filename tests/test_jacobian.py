import argparse

import pytest
import torch

from cl_ica import latent_spaces
from src.dep_mat import calc_jacobian
from src.model import ContrastiveLearningModel
from src.prob_utils import setup_marginal, setup_conditional
from src.utils import setup_seed, set_learning_mode, set_device


@pytest.fixture(params=[2, 3])
def args(request):
    import sys
    print(sys.path)
    args = argparse.Namespace(act_fct='leaky_relu', alpha=0.5, batch_size=6144, box_max=1.0, box_min=0.0, c_p=1,
                              c_param=0.05, identity_mixing_and_solution=False, inject_structure=False,
                              learnable_mask=False, load_f=None, load_g=None, lr=0.0001, m_p=0, m_param=1.0,
                              mode='unsupervised', more_unsupervised=1, n=request.param, n_eval_samples=512,
                              n_log_steps=250,
                              n_mixing_layer=3, n_steps=100001, no_cuda=False, normalization='', notes=None,
                              num_eval_batches=10, num_permutations=50, p=1, preserve_vol=False, project='experiment',
                              resume_training=False, save_dir='', seed=0, space_type='box', sphere_r=1.0, tau=1.0,
                              use_batch_norm=True, use_dep_mat=True, use_flows=True, use_reverse=False,
                              use_wandb=False, variant=1, verbose=False)

    set_device(args)
    setup_seed(args.seed)
    set_learning_mode(args)

    return args


@pytest.fixture()
def model(args):
    return ContrastiveLearningModel(args)


@pytest.mark.parametrize("network", ["decoder", "encoder"])
def test_triangularity_decoder_jacobian(model: ContrastiveLearningModel, network):
    """

    Checks the AR nature of the model by calculating the Jacobian.

    :param model: model to test
    :param network: model components
    :return:
    """

    # draw a sample from the latent space
    latent_space = latent_spaces.LatentSpace(space=model.space, sample_marginal=(setup_marginal(model.hparams)),
                                             sample_conditional=(setup_conditional(model.hparams)), )
    z = latent_space.sample_marginal(model.hparams.n_eval_samples)

    # calculate the Jacobian
    dep_mat = calc_jacobian(model._modules[network], z, normalize=model.hparams.preserve_vol).abs().mean(0)
    print(f"{dep_mat=}")

    assert (torch.tril(dep_mat) != dep_mat).sum() == 0


@pytest.mark.parametrize("network", ["decoder", "encoder"])
def test_triangularity_naive(model: ContrastiveLearningModel, network):
    """
    Checks the AR nature of the model by perturbing the input and observing the changes in the outputs.

    :param model: model to test
    :param network: model components
    :return:
    """

    # constants
    batch_size = 1
    tria_check = torch.zeros(model.hparams.n)

    # set to eval mode but remember original state
    in_training: bool = model.training
    model.eval()  # otherwise we will get 0 gradients

    # calculate the baseline output - all inputs should be different from 0
    # this is to avoid degenerate cases making the test succeed
    y0 = model._modules[network](torch.ones(batch_size, model.hparams.n))
    print(f"{y0=}")

    # loop for perturbing each input one-by-one
    for i in range(model.hparams.n):
        z = torch.ones(batch_size, model.hparams.n)
        z[:, i] = -1

        y = model._modules[network](z)

        print(f"{i=},{y=}")

        # the indexing is up to the ith element
        # as input i affects outputs i:n
        # so a change before that is a failure
        tria_check[i] = (y[:, :i] != y0[:, :i]).sum()

    # set back to original mode
    if in_training is True:
        model.train()

    assert tria_check.sum() == 0
