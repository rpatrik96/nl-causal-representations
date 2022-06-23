from pytorch_lightning import Trainer

from care_nl_ica.runner import IIAModule


def test_iia_itcl_module(itcl_module, itcl_datamodule):
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model=itcl_module, datamodule=itcl_datamodule)


def test_iia_igcl_module(igcl_datamodule):
    NUM_DATA = 2**10
    BATCH_SIZE = 64
    NET_MODEL = "igcl"
    igcl = IIAModule(net_model=NET_MODEL, num_data=NUM_DATA, batch_size=BATCH_SIZE)

    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model=igcl, datamodule=igcl_datamodule)
