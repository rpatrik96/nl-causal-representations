import numpy as np
import torch
import torch.nn as nn

from cl_ica import encoders, invertible_network_utils, losses, spaces


class ContrastiveLearningModel(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self._setup_decoder()
        self._setup_loss()
        self._setup_space()


    def _setup_encoder(self):
        hparams = self.hparams

        output_normalization, output_normalization_kwargs = configure_output_normalization(hparams)

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
        print(f"{encoder=}")

        self.encoder =  encoder


    def reset_encoder(self):
        self._setup_encoder()

    def _setup_decoder(self):
        hparams = self.hparams
        # create MLP
        ######NOTE THAT weight_matrix_init='rvs' (used in TCL data gen in icebeem) yields linear mixing!##########
        decoder = invertible_network_utils.construct_invertible_mlp(
            n=hparams.n,
            n_layers=hparams.n_mixing_layer,
            act_fct=hparams.act_fct,
            cond_thresh_ratio=0.001,
            n_iter_cond_thresh=25000,
            lower_triangular=True,
            weight_matrix_init='rvs',
            sparsity=True,
            variant=torch.from_numpy(np.array([hparams.variant]))
        )

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


def configure_output_normalization(args):
    output_normalization = None
    output_normalization_kwargs = None
    if args.normalization == "learnable_box":
        output_normalization = "learnable_box"
    elif args.normalization == "fixed_box":
        output_normalization = "fixed_box"
        output_normalization_kwargs = dict(init_abs_bound=args.box_max - args.box_min)
    elif args.normalization == "learnable_sphere":
        output_normalization = "learnable_sphere"
    elif args.normalization == "fixed_sphere":
        output_normalization = "fixed_sphere"
        output_normalization_kwargs = dict(init_r=args.sphere_r)
    elif args.normalization == "":
        print("Using no output normalization")
        output_normalization = None
    else:
        raise ValueError("Invalid output normalization:", args.normalization)
    return output_normalization, output_normalization_kwargs