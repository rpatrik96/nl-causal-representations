from typing import Optional


import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from care_nl_ica.cl_ica import invertible_network_utils
from care_nl_ica.dataset import ContrastiveDataset
from care_nl_ica.dep_mat import calc_jacobian
from care_nl_ica.graph_utils import indirect_causes
from care_nl_ica.data.sem import LinearSEM, NonLinearSEM

from care_nl_ica.utils import SpaceType, DataGenType


class ContrastiveDataModule(pl.LightningDataModule):
    def __init__(
        self,
        n_mixing_layer: int = 1,
        permute: bool = False,
        act_fct: str = "leaky_relu",
        use_sem: bool = True,
        data_gen_mode: DataGenType = "rvs",
        variant: int = 0,
        nonlin_sem: bool = False,
        box_min: float = 0.0,
        box_max: float = 1.0,
        sphere_r: float = 1.0,
        space_type: SpaceType = "box",
        m_p: int = 0,
        c_p: int = 1,
        m_param: float = 1.0,
        c_param: float = 0.05,
        batch_size: int = 64,
        latent_dim: int = 3,
        normalize_latents: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_dep_mat: bool = True,
        force_chain: bool = False,
        force_uniform: bool = False,
        diag_weight=0.0,
        offset=0,
        mask_prob=0.0,
        mlp_sparsity=False,
        **kwargs,
    ):

        """

        :param mlp_sparsity: whether the invertible MLP has a sparsity mask
        :param mask_prob: probability to delete edges in the SEM
        :param offset: constant to add to weights in SEM
        :param diag_weight: weight for adding torch.eye to the SEM weights
        :param force_chain: make the graph to a chain
        :param force_uniform: make the mixing weights the same
        :param n_mixing_layer: Number of layers in nonlinear mixing network
        :param permute: Permute causal ordering
        :param act_fct: Activation function in mixing network
        :param use_sem: Use **linear** SEM as mixing
        :param data_gen_mode: Data generation mode, can be ('rvs', 'pcl')
        :param variant: variant (e.g., ordering index)
        :param nonlin_sem: Use nonlinear SEM as mixing
        :param box_min: For box normalization only. Minimal value of box.
        :param box_max: For box normalization only. Maximal value of box.
        :param sphere_r: For sphere normalization only. Radius of the sphere.
        :param space_type: can be ("box", "sphere", "unbounded")
        :param m_p: Type of ground-truth marginal distribution. p=0 means uniform; all other p values correspond to (projected) Lp Exponential
        :param c_p: Exponent of ground-truth Lp Exponential distribution
        :param m_param: Additional parameter for the marginal (only relevant if it is not uniform)
        :param c_param: Concentration parameter of the conditional distribution
        :param batch_size: Batch size
        :param latent_dim: Dimensionality of the latents
        :param normalize_latents: Normalizes the latent (marginal) distribution
        :param device: device
        :param use_dep_mat: Use the Jacobian
        :param kwargs:
        """
        super().__init__()

        print(
            "-----------Set val_check_interval in the Trainer as the Iterable dataset does not have len!-------"
        )

        self.save_hyperparameters()

    def _setup_mixing(self):
        if self.hparams.use_sem is False:
            print("Variant is set to None")
            # create MLP
            ######NOTE THAT weight_matrix_init='rvs' (used in TCL data gen in icebeem) yields linear mixing!##########
            self.mixing = invertible_network_utils.construct_invertible_mlp(
                n=self.hparams.latent_dim,
                n_layers=self.hparams.n_mixing_layer,
                n_iter_cond_thresh=25000,
                cond_thresh_ratio=0.001,
                weight_matrix_init=self.hparams.data_gen_mode,
                act_fct=self.hparams.act_fct,
                lower_triangular=True,
                sparsity=self.hparams.mlp_sparsity,
                variant=None,
                offset=self.hparams.offset,
            )
        else:
            print("Using SEM as mixing")
            if self.hparams.nonlin_sem is False:
                self.mixing = LinearSEM(
                    num_vars=self.hparams.latent_dim,
                    permute=self.hparams.permute,
                    variant=self.hparams.variant,
                    force_chain=self.hparams.force_chain,
                    force_uniform=self.hparams.force_uniform,
                    diag_weight=self.hparams.diag_weight,
                    offset=self.hparams.offset,
                    mask_prob=self.hparams.mask_prob,
                )
            else:
                self.mixing = NonLinearSEM(
                    num_vars=self.hparams.latent_dim,
                    permute=self.hparams.permute,
                    variant=self.hparams.variant,
                    force_chain=self.hparams.force_chain,
                    force_uniform=self.hparams.force_uniform,
                    diag_weight=self.hparams.diag_weight,
                    offset=self.hparams.offset,
                    mask_prob=self.hparams.mask_prob,
                )

            # print(f"{self.mixing.weight=}")

        # make it non-trainable
        for p in self.mixing.parameters():
            p.requires_grad = False

        self.mixing = self.mixing.to(self.hparams.device)
        torch.cuda.empty_cache()

    def _calc_dep_mat(self) -> None:
        if self.hparams.use_dep_mat is True:
            # draw a sample from the latent space (marginal only)
            z = next(iter(self.train_dataloader()))[0][0, :]
            # save the decoder jacobian including the permutation
            self.mixing_jacobian_permuted = self.mixing_jacobian = (
                calc_jacobian(self.mixing, z, normalize=False).abs().max(0)[0].detach()
            )

            if self.hparams.permute is True and self.hparams.use_sem is True:
                # print(f"{dep_mat=}")
                # set_trace()
                self.mixing_jacobian = self.mixing_jacobian[
                    torch.argsort(self.mixing.permute_indices), :
                ]

            self.unmixing_jacobian = torch.tril(self.mixing_jacobian.inverse())

            self.mixing_cond = torch.linalg.cond(self.mixing_jacobian)
            self.unmixing_cond = torch.linalg.cond(self.unmixing_jacobian)

            # print(f"{self.unmixing_jacobian=}")

            self.indirect_causes, self.paths = indirect_causes(self.unmixing_jacobian)

            torch.cuda.empty_cache()

    def setup(self, stage: Optional[str] = None):
        self._setup_mixing()

        # generate data
        self.dataset = ContrastiveDataset(self.hparams, self.mixing)
        self.dl = DataLoader(self.dataset, batch_size=self.hparams.batch_size)

        self._calc_dep_mat()

    def train_dataloader(self):
        return self.dl

    def val_dataloader(self):
        return self.dl

    def test_dataloader(self):
        return self.dl

    def predict_dataloader(self):
        return self.dl

    @property
    def data_to_log(self):
        return {
            f"Mixing/mixing_jacobian": self.mixing_jacobian,
            f"Mixing/unmixing_jacobian": self.unmixing_jacobian,
            f"Mixing/mixing_cond": self.mixing_cond,
            f"Mixing/unmixing_cond": self.unmixing_cond,
            f"Mixing/indirect_causes": self.indirect_causes,
            f"Mixing/paths": self.paths,
            f"Mixing/permute_indices": torch.arange(self.hparams.latent_dim)
            if self.hparams.use_sem is False
            else self.mixing.permute_indices,
        }
