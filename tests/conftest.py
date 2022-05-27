from collections import namedtuple

import hydra.core.global_hydra
import pytest
import torch
from torch.utils.data import DataLoader

from care_nl_ica.data.datamodules import ContrastiveDataModule
from care_nl_ica.dataset import ContrastiveDataset

arg_matrix = namedtuple("arg_matrix", ["latent_dim", "use_ar_mlp"])

from hydra import compose, initialize
from pytorch_lightning import seed_everything
from argparse import Namespace


@pytest.fixture(
    params=[
        arg_matrix(latent_dim=3, use_ar_mlp=True),
        arg_matrix(latent_dim=3, use_ar_mlp=True),
    ]
)
def args(request):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../configs", job_name="test_app")

    cfg = compose(
        config_name="config",
        overrides=[
            f"data.latent_dim={request.param.latent_dim}",
            "data.use_sem=true",
            "data.nonlin_sem=true",
            "model.triangular=true",
        ],
    )

    seed_everything(cfg.seed_everything)

    return cfg


@pytest.fixture()
def dataloader(args):
    args = Namespace(
        **{**args.data, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    ds = ContrastiveDataset(
        args,
        lambda x: x
        @ torch.tril(torch.ones(args.latent_dim, args.latent_dim, device=args.device)),
    )
    dl = DataLoader(ds, args.batch_size)
    return dl


@pytest.fixture()
def datamodule(args):
    dm = ContrastiveDataModule.from_argparse_args(Namespace(**args.data))
    dm.setup()
    return dm
