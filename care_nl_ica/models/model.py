import torch
import torch.nn as nn

from care_nl_ica.cl_ica import encoders, losses
from care_nl_ica.models.sinkhorn import SinkhornNet
from strnn.models.strNN import StrNN


class ContrastiveLearningModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        if self.hparams.obs_dim is not None and self.hparams.obs_dim < 1:
            self.hparams.obs_dim = None

        self._setup_unmixing()
        self._setup_loss()

        torch.cuda.empty_cache()

    def parameters(self, recurse: bool = True):
        parameters = list(self.unmixing.parameters(recurse))

        return parameters

    def _setup_unmixing(self):
        hparams = self.hparams

        out_dim = hparams.latent_dim
        in_dim = hparams.latent_dim
        hidden_sizes = [
            hparams.latent_dim * hparams.width_factor
            for _ in range(hparams.strnn_layers)
        ]

        if hparams.strnn is False:
            (
                output_normalization,
                output_normalization_kwargs,
            ) = self._configure_output_normalization()

            encoder = encoders.get_mlp(
                n_in=in_dim,
                n_out=out_dim,
                layers=hidden_sizes,
                output_normalization=output_normalization,
                output_normalization_kwargs=output_normalization_kwargs,
            )
        else:
            adjacency = torch.tril(
                torch.ones(hparams.latent_dim, hparams.latent_dim)
            ).numpy()

            encoder = StrNN(
                nin=in_dim,
                hidden_sizes=(tuple(hidden_sizes)),
                nout=out_dim,
                opt_type="greedy",
                adjacency=adjacency,
                activation="leaky_relu",
                init_type="ian_uniform",
                norm_type="layer"
            )

            if self.hparams.permute is True:
                sinkhorn = SinkhornNet(
                    num_dim=hparams.latent_dim, num_steps=15, temperature=3e-3
                )
                encoder = nn.Sequential(
                    sinkhorn, encoder
                )  # eval needs to check causal variables to check whether the StrNN is useful

                # if re-setting the adjacency, then the weights are reinitialized

        if self.hparams.obs_dim is not None:
            obs_unmixing = []
            for _ in range(self.hparams.obs_layers - 1):
                obs_unmixing.append(
                    nn.Linear(self.hparams.obs_dim, self.hparams.obs_dim, bias=False)
                )
                obs_unmixing.append(nn.LeakyReLU(negative_slope=0.25))

            obs_unmixing.append(
                nn.Linear(self.hparams.obs_dim, self.hparams.latent_dim, bias=False)
            )
            obs_unmixing.append(nn.LeakyReLU(negative_slope=0.25))

            self.unmixing = nn.Sequential(
                *obs_unmixing,
                encoder,
            )
        else:
            self.unmixing = encoder

        if self.hparams.verbose is True:
            print(f"{self.unmixing=}")

            if (
                self.hparams.strnn is True
                and self.hparams.obs_dim is None
                and self.hparams.permute is False
            ):
                print(f"{self.unmixing[0].doubly_stochastic_matrix=}")

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
