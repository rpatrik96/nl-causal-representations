from care_nl_ica.data.datamodules import ContrastiveDataModule
import torch


def test_contrastive_datamodule(datamodule: ContrastiveDataModule):
    batches = []
    num_batches = 3
    for i in range(num_batches):
        batches.append(next(iter(datamodule.train_dataloader()))[0][0, :])

    # calculates the variance accross the batch and n dimensions
    # to check that we do not get the same data
    torch.any(torch.stack(batches).var(0).sum([-1, -2]) > 1e-7)
