from pytorch_lightning.trainer import Trainer

from care_nl_ica.data.datamodules import ContrastiveDataModule
from care_nl_ica.runner import ContrastiveICAModule


def test_runner():
    trainer = Trainer(fast_dev_run=True)
    runner = ContrastiveICAModule()
    dm = ContrastiveDataModule()
    trainer.fit(runner, datamodule=dm)
