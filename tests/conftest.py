import argparse
from collections import namedtuple

import torch
from torch.utils.data import DataLoader

from care_nl_ica.dataset import ContrastiveDataset
from care_nl_ica.args import parse_args
from care_nl_ica.datamodules import ContrastiveDataModule

import pytest

from care_nl_ica.utils import set_device, setup_seed, set_learning_mode

arg_matrix = namedtuple("arg_matrix", ["n", "use_ar_mlp"])


@pytest.fixture(
    params=[arg_matrix(n=2, use_ar_mlp=True), arg_matrix(n=3, use_ar_mlp=True)]
)
def args(request):

    args = parse_args(
        [
            "--latent_dim",
            str(request.param.n),
            "--use-ar-mlp",
            "--use-dep-mat",
            "--triangular",
            "--use-sem",
            "--nonlin-sem",
        ]
    )

    set_device(args)
    setup_seed(args.seed)
    set_learning_mode(args)

    return args


@pytest.fixture()
def dataloader(args):
    ds = ContrastiveDataset(
        args, lambda x: x @ torch.tril(torch.ones(args.n, args.n, device=args.device))
    )
    dl = DataLoader(ds, args.batch_size)
    return dl


@pytest.fixture()
def datamodule(args):
    dm = ContrastiveDataModule.from_argparse_args(args)
    dm.setup()
    return dm
