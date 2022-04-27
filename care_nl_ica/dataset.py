import torch
from torch.utils.data import Dataset

from care_nl_ica.cl_ica import latent_spaces, spaces
from care_nl_ica.prob_utils import (
    setup_marginal,
    setup_conditional,
    sample_marginal_and_conditional,
)


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


from cdt.data import load_dataset


class CDTDataset(Dataset):
    def __init__(self, hparams, dataset="sachs"):
        super().__init__()
        self.hparams = hparams

        if dataset not in (
            datasets := [
                "sachs",
                "dream4-1",
                "dream4-2",
                "dream4-3",
                "dream4-4",
                "dream4-5",
            ]
        ):
            raise ValueError(f"{dataset=}, but should be in {datasets}")

        data, graph = load_dataset(dataset)

        self.data = torch.Tensor(data.to_numpy(), device=self.hparams.device)

    def dim(self):
        return self.data.shape[1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :], self.data[idx, :]
