import argparse
from collections import namedtuple

import pytest

from care_nl_ica.utils import set_device, setup_seed, set_learning_mode

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
                              use_bias=False, l1=0.0, l2=0.0, data_gen_mode='rvs', permute=False,
                              sinkhorn=False, triangularity_loss=0.0)

    set_device(args)
    setup_seed(args.seed)
    set_learning_mode(args)

    return args