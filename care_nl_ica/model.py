import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cl_ica import encoders, invertible_network_utils, losses, spaces
from .masked_flows import MaskMAF

from care_nl_ica.mlp import ARBottleneckNet, ARMLP, LinearSEM, NonLinearSEM



class ContrastiveLearningModel(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self._setup_decoder()
        self._setup_encoder()
        self._setup_loss()
        self._setup_space()

        self._setup_learnable_jacobian()

    def parameters(self):
        parameters = list(self.encoder.parameters())

        if self.hparams.learn_jacobian is True:
            parameters += list(self.jacob.parameters())

        return parameters

    def _setup_learnable_jacobian(self):

        if self.hparams.learn_jacobian is True:
            self.jacob = LinearSEM(self.hparams.n)

            self.jacob = self.jacob.to(self.hparams.device)
            self.jacob.weight.requires_grad = True


    def _setup_encoder(self):
        hparams = self.hparams

        output_normalization, output_normalization_kwargs = self._configure_output_normalization()

        if self.hparams.use_flows is True:
            encoder = MaskMAF(hparams.n, hparams.n * 40, 5, F.relu, use_reverse=hparams.use_reverse,
                              use_batch_norm=hparams.use_batch_norm, learnable=hparams.learnable_mask)
            
            encoder.confidence.to(hparams.device)

        elif self.hparams.use_ar_mlp is True:

            encoder = ARBottleneckNet(hparams.n, [1, hparams.n * 5, hparams.n * 8, hparams.n * 12],
                                      [hparams.n * 12, hparams.n * 8, hparams.n * 5, 1], hparams.use_bias,
                                      hparams.normalization == "fixed_box", residual=True, permute=hparams.permute)

        else: 
            encoder = encoders.get_mlp(
                n_in=hparams.n,
                n_out=hparams.n,
                layers=[
                    hparams.n * 10,
                    hparams.n * 50,
                    hparams.n * 50,
                    hparams.n * 50,
                    hparams.n * 50,
                    hparams.n * 10,
                ],
                output_normalization=output_normalization,
                output_normalization_kwargs=output_normalization_kwargs
            )
        encoder = encoder.to(hparams.device)
        if hparams.load_f is not None:
            encoder.load_state_dict(torch.load(hparams.load_f, map_location=hparams.device))

        if self.hparams.verbose is True:
            print(f"{encoder=}")

        self.encoder = encoder

    @property
    def h(self):
        return ((lambda z: self.encoder(self.decoder(z))) if not self.hparams.identity_mixing_and_solution else (
            lambda z: z))

    @property
    def h_ind(self):
        return lambda z: self.decoder(z)

    def reset_encoder(self):
        self._setup_encoder()

    def _setup_decoder(self):
        hparams = self.hparams

        if hparams.use_sem is False:
            # create MLP
            ######NOTE THAT weight_matrix_init='rvs' (used in TCL data gen in icebeem) yields linear mixing!##########
            decoder = invertible_network_utils.construct_invertible_mlp(
                n=hparams.n,
                n_layers=hparams.n_mixing_layer,
                act_fct=hparams.act_fct,
                cond_thresh_ratio=0.001,
                n_iter_cond_thresh=25000,
                lower_triangular=True,
                weight_matrix_init=hparams.data_gen_mode,
                sparsity=True,
                variant=torch.from_numpy(np.array([hparams.variant]))
        )
        else:
            print("Using SEM as decoder")
            if self.hparams.nonlin_sem is False:
                decoder = LinearSEM(hparams.n)
            else:
                decoder = NonLinearSEM(hparams.n)

                
            print(f"{decoder.weight=}")

        # allocate to device
        decoder = decoder.to(hparams.device)

        # load if needed
        if hparams.load_g is not None:
            decoder.load_state_dict(torch.load(hparams.load_g, map_location=hparams.device))

        # make it non-trainable
        for p in decoder.parameters():
            p.requires_grad = False

        self.decoder = decoder

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
            self.loss = losses.SimCLRLoss(normalize=False, tau=hparams.tau, alpha=hparams.alpha)

    def _setup_space(self):
        hparams = self.hparams
        if hparams.space_type == "box":
            self.space = spaces.NBoxSpace(hparams.n, hparams.box_min, hparams.box_max)
        elif hparams.space_type == "sphere":
            self.space = spaces.NSphereSpace(hparams.n, hparams.sphere_r)
        else:
            self.space = spaces.NRealSpace(hparams.n)

    def _configure_output_normalization(self):
        hparams = self.hparams
        output_normalization = None
        output_normalization_kwargs = None
        if hparams.normalization == "learnable_box":
            output_normalization = "learnable_box"
        elif hparams.normalization == "fixed_box":
            output_normalization = "fixed_box"
            output_normalization_kwargs = dict(init_abs_bound=hparams.box_max - hparams.box_min)
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
        return self.h(x)