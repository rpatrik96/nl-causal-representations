import torch
import torch.nn as nn

from care_nl_ica.cl_ica import encoders, losses


class ContrastiveLearningModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self._setup_unmixing()
        self._setup_loss()

        torch.cuda.empty_cache()

    def parameters(self, recurse: bool = True):
        parameters = list(self.unmixing.parameters(recurse))

        return parameters

    def _setup_unmixing(self):
        hparams = self.hparams

        (
            output_normalization,
            output_normalization_kwargs,
        ) = self._configure_output_normalization()

        if self.hparams.use_ar_mlp is False:
            self.unmixing = encoders.get_mlp(
                n_in=hparams.latent_dim,
                n_out=hparams.latent_dim,
                layers=[
                    hparams.latent_dim * 10,
                    hparams.latent_dim * 50,
                    hparams.latent_dim * 50,
                    hparams.latent_dim * 50,
                    hparams.latent_dim * 50,
                    hparams.latent_dim * 10,
                ],
                output_normalization=output_normalization,
                output_normalization_kwargs=output_normalization_kwargs,
            )
        else:
            raise NotImplementedError

        if self.hparams.verbose is True:
            print(f"{self.unmixing.detach()=}")

        self.unmixing = self.unmixing.to(hparams.device)

    def _setup_loss(self):
        hparams = self.hparams

        if hparams.p:
            self.loss = losses.LpSimCLRLoss(
                p=hparams.p, tau=hparams.tau, simclr_compatibility_mode=True
            )
        else:
            self.loss = losses.SimCLRLoss(
                normalize=False, tau=hparams.tau, alpha=hparams.alpha
            )

    def _configure_output_normalization(self):
        hparams = self.hparams
        output_normalization_kwargs = None
        if hparams.normalization == "learnable_box":
            output_normalization = "learnable_box"
        elif hparams.normalization == "fixed_box":
            output_normalization = "fixed_box"
            output_normalization_kwargs = dict(
                init_abs_bound=hparams.box_max - hparams.box_min
            )
        elif hparams.normalization == "learnable_sphere":
            output_normalization = "learnable_sphere"
        elif hparams.normalization == "fixed_sphere":
            output_normalization = "fixed_sphere"
            output_normalization_kwargs = dict(init_r=hparams.sphere_r)
        elif hparams.normalization == "":
            print("Using no output normalization")
            output_normalization = None
        else:
            raise ValueError("Invalid output normalization:", hparams.normalization)
        return output_normalization, output_normalization_kwargs

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            return tuple(map(self.unmixing, x))
        else:
            return self.unmixing(x)
