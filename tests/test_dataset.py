import torch
from torch.utils.data import DataLoader

from care_nl_ica.dataset import ContrastiveDataset


def test_contrastive_dataset(args):
    ds = ContrastiveDataset(args)
    dl = DataLoader(ds, args.batch_size)

    batches = []
    num_batches = 3
    for i in range(num_batches):
        batches.append(torch.stack(list(dl)))

    # calculates the variance accross the batch and n dimensions
    # to check that we do not get the same data
    torch.any(torch.cat(batches).var(0).sum([-1, -2]) < 1e-7)
