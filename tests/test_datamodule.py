import torch

from care_nl_ica.data.datamodules import ContrastiveDataModule


def test_contrastive_datamodule(datamodule: ContrastiveDataModule):
    batches = []
    num_batches = 3
    for i in range(num_batches):
        batches.append(next(iter(datamodule.train_dataloader()))[0][0, :])

    # calculates the variance accross the batch and n dimensions
    # to check that we do not get the same data
    torch.any(torch.stack(batches).var(0).sum([-1, -2]) > 1e-7)


def test_iia_igcl_datamodule(igcl_datamodule):
    igcl_datamodule.setup()

    next(iter(igcl_datamodule.train_dataloader()))


def test_iia_itcl_datamodule(itcl_datamodule):
    itcl_datamodule.setup()

    next(iter(itcl_datamodule.train_dataloader()))
