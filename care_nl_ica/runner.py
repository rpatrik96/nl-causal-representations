import pytorch_lightning as pl
import torch
import wandb

from care_nl_ica.dep_mat import jacobians
from care_nl_ica.losses.utils import Losses
from care_nl_ica.metrics.dep_mat import (
    extract_permutation_from_jacobian,
    permutation_loss,
)
from care_nl_ica.metrics.ica_dis import (
    calc_disent_metrics,
    DisentanglementMetrics,
    frobenius_diagonality,
    corr_matrix,
)
from care_nl_ica.models.model import ContrastiveLearningModel
from care_nl_ica.utils import matrix_to_dict


class ContrastiveICAModule(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        latent_dim: int = 3,
        use_ar_mlp: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        qr: float = 0.0,
        triangularity_loss: float = 0.0,
        entropy: float = 0.0,
        permute: bool = False,
        l1: float = 0.0,
        budget: float = 0.0,
        use_reverse: bool = False,
        use_batch_norm: bool = False,
        learnable_mask: bool = False,
        sinkhorn: bool = False,
        triangular: bool = False,
        verbose: bool = False,
        p: float = 1,
        tau: float = 1.0,
        alpha: float = 0.5,
        box_min: float = 0.0,
        box_max: float = 1.0,
        sphere_r: float = 1.0,
        normalization: str = "",
        start_step=None,
        cholesky_permutation: bool = False,
        use_flows=False,
        use_bias=False,
        normalize_latents: bool = True,
        log_latent_rec=False,
    ):
        """

        :param log_latent_rec:
        :param normalize_latents:
        :param use_bias:
        :param use_flows:
        :param lr:
        :param latent_dim:
        :param use_ar_mlp:
        :param device:
        :param qr:
        :param triangularity_loss:
        :param entropy:
        :param permute:
        :param l1:
        :param budget:
        :param use_reverse:
        :param use_batch_norm:
        :param learnable_mask:
        :param sinkhorn:
        :param triangular:
        :param verbose:
        :param p:
        :param tau:
        :param alpha:
        :param box_min:
        :param box_max:
        :param sphere_r:
        :param normalization:
        :param start_step:
        """
        super().__init__()
        self.save_hyperparameters()

        self.model: ContrastiveLearningModel = ContrastiveLearningModel(self.hparams)

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            self.logger.watch(self.model, log="all", log_freq=250)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        panel_name = "Train"
        _, _, _, losses = self._forward(batch)
        self.log(f"{panel_name}/losses", losses.log_dict())

        return losses.total_loss

    def validation_step(self, batch, batch_idx):
        panel_name = "Val"
        sources, mixtures, reconstructions, losses = self._forward(batch)
        self.log(f"{panel_name}/losses", losses.log_dict())

        dep_mat = self._calc_and_log_matrices(mixtures, sources)

        disent_metrics: DisentanglementMetrics = calc_disent_metrics(
            sources[0], reconstructions[0]
        )

        self.log(f"{panel_name}/disent", disent_metrics.log_dict())

        # Update the metrics
        # todo: integrate torchmetrics

        # jacobian_metrics: JacobianMetrics = calc_jacobian_metrics(
        #     dep_mat,
        #     self.gt_jacobian_encoder,
        #     self.indirect_causes,
        #     self.gt_jacobian_decoder_permuted,
        #     threshold=3e-5,
        # )
        # self.log(jacobian_metrics.log_dict(panel_name))

        self.log_scatter_latent_rec(sources[0], reconstructions[0], "n1")
        self.log_scatter_latent_rec(mixtures[0], reconstructions[0], "z1_n1_rec")

        return losses.total_loss

    def _calc_and_log_matrices(self, mixtures, sources):
        dep_mat, numerical_jacobian, enc_dec_jac = jacobians(
            self.model, sources[0], mixtures[0]
        )

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            self.logger.experiment.log({"Unmixing/unmixing_jacobian": dep_mat.detach()})

        if self.hparams.verbose is True:
            # log the Jacobian
            matrix_to_dict(dep_mat, "a", "Encoder Jacobian")

            # log the Encoder-Decoder Jacobian
            matrix_to_dict(enc_dec_jac, "j", "Encoder-Decoder Jacobian")

            # log the numerical Jacobian
            matrix_to_dict(numerical_jacobian, "a_num", "Numerical Encoder Jacobian")

            # log the bottleneck weights
            if hasattr(self.model.unmixing, "ar_bottleneck") is True:
                matrix_to_dict(
                    self.model.unmixing.ar_bottleneck,
                    "w",
                    "AR Bottleneck Weights",
                    triangular=False,
                )

            # log the Sinkhorn matrix
            matrix_to_dict(
                self.model.sinkhorn.doubly_stochastic_matrix, "sink", "Sinkhorn matrix"
            )
        return dep_mat

    def _forward(self, batch):
        sources, mixtures = batch
        sources, mixtures = tuple(sources), tuple(mixtures)
        # forward
        reconstructions = self.model(mixtures)
        _, _, [loss_pos_mean, loss_neg_mean] = self.model.loss(
            *sources, *reconstructions
        )
        # set_trace()

        losses = Losses(
            cl_pos=loss_pos_mean,
            cl_neg=loss_neg_mean,
            sinkhorn_entropy=self.model.sinkhorn_entropy_loss,
            bottleneck_l1=self.model.bottleneck_l1_loss,
            sparsity_budget=self.model.budget_loss,
            triangularity=self.triangularity_loss(*sources, *reconstructions),
            qr=self.qr,
        )

        return sources, mixtures, reconstructions, losses

    @property
    def qr(self):

        loss = 0.0
        if self.hparams.qr != 0.0 and (
            self.hparams.start_step is None
            or (
                self.hparams.start_step is not None
                and self.global_step >= self.hparams.start_step
            )
        ):

            if self.dep_mat is not None:

                if self.hparams.use_ar_mlp is False:
                    J = self.dep_mat
                else:
                    if self.hparams.sinkhorn is False:
                        J = self.model.unmixing.ar_bottleneck.assembled_weight
                    else:
                        J = (
                            self.model.unmixing.ar_bottleneck.assembled_weight
                            @ self.model.sinkhorn.doubly_stochastic_matrix
                        )

                # Q is incentivized to be the permutation for the causal ordering
                Q = extract_permutation_from_jacobian(
                    J, self.hparams.cholesky_permutation is False
                )

                """
                The first step is to ensure that the Q in the QR decomposition of the transposed(bottleneck) is 
                **a permutation** matrix.

                The second step is to ensure that the permutation matrix is the identity. If we got a permutation matrix
                in the first step, then we could use Q.T to multiply the observations. 
                """

                # loss options
                if self.logger.global_step % 250 == 0:
                    print(f"{Q=}")

                loss = self.hparams.qr * permutation_loss(Q, matrix_power=False)

        return loss

    def triangularity_loss(self, n1, n2_con_n1, n3, n1_rec, n2_con_n1_rec, n3_rec):

        loss = 0.0
        if self.hparams.triangularity_loss != 0.0 and (
            self.hparams.start_step is None
            or (
                self.hparams.start_step is not None
                and self.logger.global_step >= self.hparams.start_step
            )
        ):
            pearson_n1 = corr_matrix(self.model.decoder(n1).T, n1_rec.T)
            pearson_n2_con_n1 = corr_matrix(
                self.model.decoder(n2_con_n1).T, n2_con_n1_rec.T
            )
            pearson_n3 = corr_matrix(self.model.decoder(n3).T, n3_rec.T)

            loss = (
                self.hparams.triangularity_loss
                * (
                    frobenius_diagonality(pearson_n1.abs())
                    + frobenius_diagonality(pearson_n2_con_n1.abs())
                    + frobenius_diagonality(pearson_n3.abs())
                ).mean()
            )

        return loss

    def log_scatter_latent_rec(self, latent, rec, name: str):

        if (
            self.hparams.log_latent_rec is True
            and isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True
        ):
            for i in range(self.hparams.latent_dim):
                table = wandb.Table(
                    data=torch.stack((latent[:, i], rec[:, i])).T.tolist(),
                    columns=["latent", "rec"],
                )

                self.logger.experiment.log(
                    {
                        f"latent_rec_{name}_dim_{i}": wandb.plot.scatter(
                            table,
                            "latent",
                            "rec",
                            title=f"Latents vs reconstruction of {name} in dimension {i}",
                        )
                    }
                )

    def on_fit_start(self) -> None:
        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            self.logger.experiment.log(self.trainer.datamodule.data_to_log)

    def on_fit_end(self) -> None:
        pass
