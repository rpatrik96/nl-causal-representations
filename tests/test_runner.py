from pytorch_lightning.trainer import Trainer, seed_everything

from care_nl_ica.data.datamodules import ContrastiveDataModule
from care_nl_ica.runner import ContrastiveICAModule

import pytest


# parametrize by both strnn flag and obs_dim
@pytest.mark.parametrize(
    "obs_dim,strnn", [(None, False), (None, True), (5, False), (5, True)]
)
def test_runner(obs_dim, strnn):
    seed_everything(42)
    trainer = Trainer(fast_dev_run=True)
    batch_size = 16
    runner = ContrastiveICAModule(
        strnn=strnn, obs_dim=obs_dim, strnn_layers=2, obs_layers=2
    )
    dm = ContrastiveDataModule(batch_size=batch_size, obs_dim=obs_dim)
    trainer.fit(runner, datamodule=dm)
