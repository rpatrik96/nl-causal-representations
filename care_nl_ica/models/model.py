import torch
import torch.nn as nn
import torch.nn.functional as F

from care_nl_ica.cl_ica import encoders, losses
from care_nl_ica.models.masked_flows import MaskMAF
from care_nl_ica.models.mlp import ARBottleneckNet


class ContrastiveLearningModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self._setup_unmixing()
        self._setup_loss()

        self.sinkhorn_net = None  # SinkhornNet(hparams.latent_dim, 15, 1e-3)
        if self.sinkhorn_net is not None:
            print("Using model-level sinkhorn")
            self.sinkhorn_net.to(hparams.device)

    def parameters(self, recurse: bool = True):
        parameters = list(self.unmixing.parameters(recurse))

        if self.sinkhorn_net is not None:
            parameters += list(self.sinkhorn_net.parameters(recurse))

        return parameters

    @property
    def sinkhorn(self):
        if self.sinkhorn_net is not None:
            sinkhorn = self.sinkhorn_net
        elif self.hparams.use_ar_mlp is True:
            sinkhorn = self.unmixing.sinkhorn
        elif self.hparams.use_ar_mlp is False:
            sinkhorn = self.unmixing[0]
        else:
            sinkhorn = None

        return sinkhorn

    def sinkhorn_entropy(self):
        probs = F.softmax(self.sinkhorn.doubly_stochastic_matrix, -1).view(
            -1,
        )
        return torch.distributions.Categorical(probs).entropy()

    def sinkhorn_entropy_loss(self):
        loss = 0
        if self.hparams.entropy != 0.0 and self.hparams.sinkhorn is True:
            loss = self.hparams.entropy * self.sinkhorn_entropy()

        return loss

    def bottleneck_l1_loss(self):
        loss = 0
        if self.hparams.l1 != 0 and self.hparams.use_ar_mlp is True:
            # add sparsity loss to the AR MLP bottleneck
            loss = self.hparams.l1 * self.unmixing.bottleneck_l1_norm

        return loss

    def budget_loss(self):
        loss = 0
        if self.hparams.budget != 0.0 and self.hparams.use_ar_mlp is True:
            loss = (
                self.hparams.budget * self.unmixing.ar_bottleneck.budget_net.budget_loss
            )

            if self.hparams.entropy != 0.0:
                loss = (
                    self.hparams.entropy
                    * self.unmixing.ar_bottleneck.budget_net.entropy
                )
        return loss

    def _setup_unmixing(self):
        hparams = self.hparams

        (
            output_normalization,
            output_normalization_kwargs,
        ) = self._configure_output_normalization()

        if self.hparams.use_flows is True:
            unmixing = MaskMAF(
                hparams.latent_dim,
                hparams.latent_dim * 40,
                5,
                F.relu,
                use_reverse=hparams.use_reverse,
                use_batch_norm=hparams.use_batch_norm,
                learnable=hparams.learnable_mask,
            )

            unmixing.confidence.to(hparams.device)

        elif self.hparams.use_ar_mlp is True:

            unmixing = ARBottleneckNet(
                hparams.latent_dim,
                [
                    1,
                    hparams.latent_dim * 10,
                    hparams.latent_dim * 20,
                    hparams.latent_dim * 20,
                ],
                [
                    hparams.latent_dim * 20,
                    hparams.latent_dim * 20,
                    hparams.latent_dim * 10,
                    1,
                ],
                hparams.use_bias,
                hparams.normalization == "fixed_box",
                residual=False,
                sinkhorn=hparams.sinkhorn,
                triangular=self.hparams.triangular,
                budget=(self.hparams.budget != 0.0),
            )

        else:
            unmixing = encoders.get_mlp(
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
                sinkhorn=hparams.sinkhorn,
            )

        if self.hparams.verbose is True:
            print(f"{unmixing=}")

        self.unmixing = unmixing.to(hparams.device)

    def _setup_loss(self):
        hparams = self.hparams

        if hparams.p:
            """
            loss = losses.LpSimCLRLoss(
                p=args.p, tau=args.tau, simclr_compatibility_mode=False, alpha=args.alpha, simclr_denominator=True
            )
            """
            """
            loss = losses.LpSimCLRLoss(
                p=args.p, tau=args.tau, simclr_compatibility_mode=True, alpha=args.alpha, simclr_denominator=False
            )
            """
            self.loss = losses.LpSimCLRLoss(
                p=hparams.p, tau=hparams.tau, simclr_compatibility_mode=True
            )
        else:
            self.loss = losses.SimCLRLoss(
                normalize=False, tau=hparams.tau, alpha=hparams.alpha
            )

    def _configure_output_normalization(self):
        hparams = self.hparams
        output_normalization = None
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
