from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from IIA.subfunc.generate_artificial_data import generate_artificial_data, apply_mlp
from IIA.subfunc.preprocessing import pca
from care_nl_ica.cl_ica import invertible_network_utils
from care_nl_ica.data.sem import LinearSEM, NonLinearSEM
from care_nl_ica.dataset import ContrastiveDataset, ConditionalDataset
from care_nl_ica.dep_mat import calc_jacobian
from care_nl_ica.graph_utils import indirect_causes
from care_nl_ica.utils import SpaceType, DataGenType


class AugmentedNVARModel(nn.Module):
    def __init__(self, mlplayers, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope
        self.seq = self._setup(mlplayers)

    def _setup(self, mlplayers):
        """Convert MLP paramneters into a torch model to calculate the Jacobian"""
        mixing_layers = []
        for mlplayer in mlplayers:
            _, A, b = mlplayer.values()
            # A = A.T

            # need to separate bias from the linear layer as
            # the original code adds bias before multiplying with the weights
            bias = BiasNet(A.shape[1])
            bias.bias = nn.Parameter(
                torch.from_numpy(b.astype(np.float32)).reshape(1, -1),
                requires_grad=False,
            )
            mixing_layers.append(bias)

            lin = nn.Linear(A.shape[0], A.shape[1], bias=False)
            lin.weight = nn.Parameter(
                torch.from_numpy(A.astype(np.float32)), requires_grad=False
            )
            mixing_layers.append(lin)

            mixing_layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
        # last layer has no ReLU
        mixing_layers.pop()
        return nn.Sequential(*mixing_layers)

    def forward(self, input):
        return torch.concat([self.seq(input), input[:, input.shape[1] // 2 :]], 1)


class BiasNet(nn.Module):
    def __init__(self, dim):

        super().__init__()
        self.dim = dim

        self.bias = nn.Parameter(torch.empty(self.dim))

    def forward(self, input):
        return input + self.bias


class IIADataModule(pl.LightningDataModule):
    def __init__(
        self
        # Data generation ---------------------------------------------
        ,
        num_layer=3,  # number of layers of mixing-MLP
        num_comp=20,  # number of components (dimension)
        num_data=2**18,  # number of data points
        num_data_test=2**18,  # number of data points
        num_basis=64,  # number of frequencies of fourier bases
        modulate_range=[-2, 2],
        modulate_range2=[-2, 2],
        ar_order=1,
        random_seed=0,  # random seed
        net_model="itcl",  # "itcl" or "igcl"
        num_segment=256,  # None for IGCL
        train_ratio=0.8,
        negative_slope=0.2,
        batch_size=512,
        mix_mode="dyn",
    ):
        super().__init__()

        self.save_hyperparameters()

        self.hparams.cat_input = self.hparams.mix_mode == "dyn"

        self._generate_data()

    def _generate_data(self):
        # Generate sensor signal --------------------------------------
        x, s, y, x_test, s_test, y_test, mlplayers, _, _ = generate_artificial_data(
            num_comp=self.hparams.num_comp,
            num_data=self.hparams.num_data,
            num_data_test=self.hparams.num_data_test,
            num_layer=self.hparams.num_layer,
            num_basis=self.hparams.num_basis,
            modulate_range1=self.hparams.modulate_range,
            modulate_range2=self.hparams.modulate_range2,
            random_seed=self.hparams.random_seed,
            mix_mode=self.hparams.mix_mode,
        )

        if self.hparams.net_model == "itcl":  # Remake label for TCL learning
            num_segmentdata = int(
                np.ceil(self.hparams.num_data / self.hparams.num_segment)
            )
            y = np.tile(
                np.arange(self.hparams.num_segment), [num_segmentdata, 1]
            ).T.reshape(-1)[: self.hparams.num_data]
        # Preprocessing -----------------------------------------------
        x, self.hparams.pca_parm = pca(x, num_comp=self.hparams.num_comp)  # PCA

        return x, y, s, x_test, y_test, s_test, mlplayers

    def setup(self, stage: Optional[str] = None):
        (
            obs,
            labels,
            sources,
            obs_test,
            labels_test,
            sources_test,
            mlplayers,
        ) = self._generate_data()

        self.mixing = AugmentedNVARModel(mlplayers, self.hparams.negative_slope)

        # generate data
        tr_val_dataset = ConditionalDataset(
            obs,
            labels,
            sources,
            batch_size=self.hparams.batch_size,
            ar_order=self.hparams.ar_order,
            transform=None
            # the mixing is already applied in `generate_artificial_data`
        )

        # split
        train_len = int(self.hparams.train_ratio * len(tr_val_dataset))
        val_len = int(len(tr_val_dataset) - train_len)
        self.ds_train, self.ds_val = random_split(tr_val_dataset, [train_len, val_len])

        self.ds_test_pred = ConditionalDataset(
            obs_test,
            labels_test,
            sources_test,
            batch_size=self.hparams.batch_size,
            ar_order=self.hparams.ar_order,
            transform=None
            # the mixing is already applied in `generate_artificial_data`
        )
        # dataloaders
        self.dl_train = DataLoader(self.ds_train, batch_size=1)
        self.dl_val = DataLoader(self.ds_val, batch_size=1)
        self.dl_test_pred = DataLoader(self.ds_test_pred, batch_size=1)

        self._calc_dep_mat()

    def _calc_dep_mat(self) -> None:
        # draw a sample from the latent space (marginal only)
        sources = self.train_dataloader().dataset.dataset.sources[
            : self.hparams.batch_size
        ]
        obs = torch.ones_like(sources)
        if self.hparams.cat_input is True:
            mixing_input = torch.concat([sources, obs], 1)

        # save the decoder jacobian
        self.mixing_jacobian = (
            calc_jacobian(self.mixing, mixing_input, normalize=False)
            .abs()
            .mean(0)
            .detach()
        )

        self.unmixing_jacobian = torch.tril(self.mixing_jacobian.inverse())

        self.mixing_cond = torch.linalg.cond(self.mixing_jacobian)
        self.unmixing_cond = torch.linalg.cond(self.unmixing_jacobian)

        self.indirect_causes, self.paths = indirect_causes(self.unmixing_jacobian)

        torch.cuda.empty_cache()

    def train_dataloader(self):
        return self.dl_train

    def val_dataloader(self):
        return self.dl_val

    def test_dataloader(self):
        return self.dl_test_pred

    def predict_dataloader(self):
        return self.dl_test_pred

    @property
    def data_to_log(self):
        return {
            f"Mixing/mixing_jacobian": self.mixing_jacobian,
            f"Mixing/unmixing_jacobian": self.unmixing_jacobian,
            f"Mixing/mixing_cond": self.mixing_cond,
            f"Mixing/unmixing_cond": self.unmixing_cond,
            f"Mixing/indirect_causes": self.indirect_causes,
            f"Mixing/paths": self.paths,
        }


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
                calc_jacobian(self.mixing, z, normalize=False).abs().mean(0).detach()
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
