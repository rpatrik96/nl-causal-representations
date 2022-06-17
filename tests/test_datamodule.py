from care_nl_ica.data.datamodules import ContrastiveDataModule, IIADataModule
import torch


def test_contrastive_datamodule(datamodule: ContrastiveDataModule):
    batches = []
    num_batches = 3
    for i in range(num_batches):
        batches.append(next(iter(datamodule.train_dataloader()))[0][0, :])

    # calculates the variance accross the batch and n dimensions
    # to check that we do not get the same data
    torch.any(torch.stack(batches).var(0).sum([-1, -2]) > 1e-7)


def test_iia_igcl_datamodule():

    NUM_DATA = 2**10
    datamodule = IIADataModule(
        num_data=NUM_DATA, num_data_test=NUM_DATA, net_model="igcl", batch_size=64
    )
    datamodule.setup()

    next(iter(datamodule.train_dataloader()))


def test_iia_itcl_datamodule():
    NUM_DATA = 2**10
    datamodule = IIADataModule(
        num_data=NUM_DATA, num_data_test=NUM_DATA, net_model="itcl", batch_size=64
    )
    datamodule.setup()

    next(iter(datamodule.train_dataloader()))
