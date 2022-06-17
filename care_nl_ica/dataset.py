import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset

from care_nl_ica.cl_ica import latent_spaces, spaces
from care_nl_ica.prob_utils import (
    setup_marginal,
    setup_conditional,
    sample_marginal_and_conditional,
)


class ConditionalDataset(Dataset):
    """
    a Dataset object holding a tuple (x,y): observed and auxiliary variable
    """

    def __init__(
        self, obs, labels, sources, batch_size=512, transform=None, ar_order=1
    ):
        self.transform = transform
        self.batch_size = batch_size
        self.ar_order = ar_order

        self.sources = torch.from_numpy(sources.T.astype(np.float32))
        self.obs = torch.from_numpy(obs.T.astype(np.float32))
        self.labels = torch.from_numpy(labels).type(torch.LongTensor)

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, index):
        # make shuffled batch
        t_idx = (
            np.random.permutation(self.obs.shape[0] - self.ar_order)[: self.batch_size]
            + self.ar_order
        )
        t_idx_ar = t_idx.reshape([-1, 1]) + np.arange(
            0, -self.ar_order - 1, -1
        ).reshape([1, -1])
        obs = self.obs[t_idx_ar.reshape(-1), :].reshape(
            [self.batch_size, self.ar_order + 1, -1]
        )
        labels = self.labels[t_idx]
        sources = self.sources[t_idx]

        # only for IGCL (y_torch in the original repo)
        true_logits = torch.cat(
            [torch.ones([self.batch_size]), torch.zeros([self.batch_size])]
        )

        if self.transform is not None:
            obs = self.transform(obs)
        return obs, labels, sources, true_logits


class ContrastiveDataset(torch.utils.data.IterableDataset):
    def __init__(self, hparams, transform=None):
        super().__init__()
        self.hparams = hparams
        self.transform = transform

        self._setup_space()

        self.latent_space = latent_spaces.LatentSpace(
            space=self.space,
            sample_marginal=setup_marginal(self.hparams),
            sample_conditional=setup_conditional(self.hparams),
        )
        torch.cuda.empty_cache()

    def _setup_space(self):
        if self.hparams.space_type == "box":
            self.space = spaces.NBoxSpace(
                self.hparams.latent_dim, self.hparams.box_min, self.hparams.box_max
            )
        elif self.hparams.space_type == "sphere":
            self.space = spaces.NSphereSpace(
                self.hparams.latent_dim, self.hparams.sphere_r
            )
        else:
            self.space = spaces.NRealSpace(self.hparams.latent_dim)

    def __iter__(self):
        sources = torch.stack(
            sample_marginal_and_conditional(
                self.latent_space,
                size=self.hparams.batch_size,
                device=self.hparams.device,
            )
        )

        mixtures = torch.stack(tuple(map(self.transform, sources)))
        return iter((sources, mixtures))


# from cdt.data import load_dataset
#
#
# class CDTDataset(Dataset):
#     def __init__(self, hparams, dataset="sachs"):
#         super().__init__()
#         self.hparams = hparams
#
#         if dataset not in (
#             datasets := [
#                 "sachs",
#                 "dream4-1",
#                 "dream4-2",
#                 "dream4-3",
#                 "dream4-4",
#                 "dream4-5",
#             ]
#         ):
#             raise ValueError(f"{dataset=}, but should be in {datasets}")
#
#         data, graph = load_dataset(dataset)
#
#         self.data = torch.Tensor(data.to_numpy(), device=self.hparams.device)
#
#     def dim(self):
#         return self.data.shape[1]
#
#     def __len__(self):
#         return self.data.shape[0]
#
#     def __getitem__(self, idx):
#         return self.data[idx, :], self.data[idx, :]
