import argparse
from collections import namedtuple

import pytest
import torch
from torch.autograd.functional import jacobian

from care_nl_ica.cl_ica import latent_spaces
from care_nl_ica.dep_mat import calc_jacobian, calc_jacobian_numerical
from care_nl_ica.model import ContrastiveLearningModel
from care_nl_ica.prob_utils import setup_marginal, setup_conditional
from care_nl_ica.utils import setup_seed, set_learning_mode, set_device

arg_matrix = namedtuple('arg_matrix', ['n', 'use_ar_mlp'])


@pytest.fixture(params=[arg_matrix(n=2, use_ar_mlp=False),
                        arg_matrix(n=2, use_ar_mlp=True),
                        arg_matrix(n=3, use_ar_mlp=False),
                        arg_matrix(n=3, use_ar_mlp=True)])
def args(request):
    args = argparse.Namespace(act_fct='leaky_relu', alpha=0.5, batch_size=6144, box_max=1.0, box_min=0.0, c_p=1,
                              c_param=0.05, identity_mixing_and_solution=False, inject_structure=False,
                              learnable_mask=False, load_f=None, load_g=None, lr=0.0001, m_p=0, m_param=1.0,
                              mode='unsupervised', more_unsupervised=1, n=request.param.n, n_eval_samples=512,
                              n_log_steps=250, n_mixing_layer=3, n_steps=100001, no_cuda=False, normalization='',
                              notes=None, num_eval_batches=10, num_permutations=50, p=1, normalize_latents=False,
                              project='experiment', resume_training=False, save_dir='', seed=0, space_type='box',
                              sphere_r=1.0, tau=1.0, use_batch_norm=True, use_dep_mat=True,
                              use_flows=not request.param.use_ar_mlp, use_reverse=False, use_wandb=False, variant=1,
                              verbose=False, use_ar_mlp=request.param.use_ar_mlp, use_sem=True, nonlin_sem=False,
                              use_bias=False, l1=0.0, l2=0.0, data_gen_mode='rvs', learn_jacobian=False, permute=False, sinkhorn=False)

    set_device(args)
    setup_seed(args.seed)
    set_learning_mode(args)

    return args


@pytest.fixture()
def model(args):
    return ContrastiveLearningModel(args)


@pytest.mark.parametrize("network", ["decoder", "encoder"])
def test_triangularity_jacobian(model: ContrastiveLearningModel, network, numerical_check: bool = False,
                                built_in_jacobian_check: bool = False):
    """

    Checks the AR nature of the model by calculating the Jacobian.

    :param model: model to test
    :param network: model components
    :return:
    """

    print("\n------------------------")
    print(f"{network=}")
    print("------------------------")

    # draw a sample from the latent space
    latent_space = latent_spaces.LatentSpace(space=model.space, sample_marginal=(setup_marginal(model.hparams)),
                                             sample_conditional=(setup_conditional(model.hparams)), )
    z = latent_space.sample_marginal(model.hparams.n_eval_samples)

    # calculate the Jacobian
    dep_mat = calc_jacobian(model._modules[network], z, normalize=model.hparams.normalize_latents).mean(0)
    print(f"{dep_mat=}")

    # numerical Jacobian
    if numerical_check is True:
        print(f"{calc_jacobian_numerical(model._modules[network], z, model.hparams.n, model.hparams.device)=}")

    # same as calc_jacobian, but using the torch jacobian function
    if built_in_jacobian_check is True:
        # x in shape (Batch, Length)
        def _func_sum(x):
            return model._modules[network].forward(x).sum(dim=0)

        print("---------------")

        print(jacobian(_func_sum, z).permute(1, 0, 2).abs().mean(0))

    assert (torch.tril(dep_mat) != dep_mat).sum() == 0


@pytest.mark.parametrize("network", ["decoder", "encoder"])
def test_triangularity_naive(model: ContrastiveLearningModel, network):
    """
    Checks the AR nature of the model by perturbing the input and observing the changes in the outputs.

    :param model: model to test
    :param network: model components
    :return:
    """

    print("------------------------")
    print(f"{network=}")
    print("------------------------")

    # constants
    batch_size = 1
    tria_check = torch.zeros(model.hparams.n)

    # set to eval mode but remember original state
    in_training: bool = model.training
    model.eval()  # otherwise we will get 0 gradients

    # calculate the baseline output - all inputs should be different from 0
    # this is to avoid degenerate cases making the test succeed
    y0 = model._modules[network](torch.ones(batch_size, model.hparams.n).to(model.hparams.device))
    print(f"{y0=}")

    # unsqueeze for the AR MLP
    if len(y0.shape) == 1:
        y0 = y0.unsqueeze(0)

    # loop for perturbing each input one-by-one
    for i in range(model.hparams.n):
        z = torch.ones(batch_size, model.hparams.n).to(model.hparams.device)
        z[:, i] = -1

        y = model._modules[network](z)

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
