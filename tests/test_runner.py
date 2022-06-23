from pytorch_lightning import Trainer

from care_nl_ica.data.datamodules import IIADataModule
from care_nl_ica.runner import IIAModule


def test_iia_itcl_module():
    NUM_DATA = 2**10
    BATCH_SIZE = 64
    NET_MODEL = "itcl"
    itcl = IIAModule(net_model=NET_MODEL, num_data=NUM_DATA, batch_size=BATCH_SIZE)

    dm = IIADataModule(
        num_data=NUM_DATA,
        num_data_test=NUM_DATA,
        net_model=NET_MODEL,
        batch_size=BATCH_SIZE,
    )

    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model=itcl, datamodule=dm)


def test_iia_igcl_module():
    NUM_DATA = 2**10
    BATCH_SIZE = 64
    NET_MODEL = "igcl"
    igcl = IIAModule(net_model=NET_MODEL, num_data=NUM_DATA, batch_size=BATCH_SIZE)

    dm = IIADataModule(
        num_data=NUM_DATA,
        num_data_test=NUM_DATA,
        net_model=NET_MODEL,
        batch_size=BATCH_SIZE,
    )

    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model=igcl, datamodule=dm)
