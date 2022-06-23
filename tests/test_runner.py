from pytorch_lightning import Trainer


def test_iia_itcl_module(itcl_module, itcl_datamodule):
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model=itcl_module, datamodule=itcl_datamodule)


def test_iia_igcl_module(igcl_module, igcl_datamodule):
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model=igcl_module, datamodule=igcl_datamodule)
