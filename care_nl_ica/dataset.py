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

    def _setup_space(self):
        hparams = self.hparams
        if hparams.space_type == "box":
            self.space = spaces.NBoxSpace(
                hparams.latent_dim, hparams.box_min, hparams.box_max
            )
        elif hparams.space_type == "sphere":
            self.space = spaces.NSphereSpace(hparams.latent_dim, hparams.sphere_r)
        else:
            self.space = spaces.NRealSpace(hparams.latent_dim)

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
