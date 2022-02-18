import torch


def test_contrastive_dataset(dataloader):
    batches = []
    num_batches = 3
    for i in range(num_batches):
        batches.append(next(iter(dataloader)))

    # calculates the variance accross the batch and n dimensions
    # to check that we do not get the same data
    torch.any(torch.cat(batches).var(0).sum([-1, -2]) < 1e-7)
