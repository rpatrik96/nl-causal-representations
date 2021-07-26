import argparse

import pytest
import torch

from cl_ica import latent_spaces
from dep_mat import calc_jacobian
from model import ContrastiveLearningModel
from prob_utils import setup_marginal, setup_conditional
from utils import setup_seed, set_learning_mode, set_device


@pytest.fixture(params=[2,3])
def args(request):
    args = argparse.Namespace(act_fct='leaky_relu', alpha=0.5, batch_size=6144, box_max=1.0, box_min=0.0, c_p=1,
                              c_param=0.05, identity_mixing_and_solution=False, inject_structure=False,
                              learnable_mask=False, load_f=None, load_g=None, lr=0.0001, m_p=0, m_param=1.0,
                              mode='unsupervised', more_unsupervised=1, n=request.param, n_eval_samples=512, n_log_steps=250,
                              n_mixing_layer=3, n_steps=100001, no_cuda=False, normalization='', notes=None,
                              num_eval_batches=10, num_permutations=50, p=1, preserve_vol=False, project='experiment',
                              resume_training=False, save_dir='', seed=0, space_type='box', sphere_r=1.0, tau=1.0,
                              use_batch_norm=False, use_dep_mat=True, use_flows=True, use_reverse=False,
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
    latent_space = latent_spaces.LatentSpace(space=model.space, sample_marginal=(setup_marginal(model.hparams)),
                                             sample_conditional=(setup_conditional(model.hparams)), )

    z = latent_space.sample_marginal(model.hparams.n_eval_samples)
    dep_mat = calc_jacobian(model._modules[network], z, normalize=model.hparams.preserve_vol).abs().mean(0)

    print(f"{dep_mat=}")

    assert (torch.tril(dep_mat) != dep_mat).sum() == 0

