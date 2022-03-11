import sys

import pytorch_lightning as pl
import torch
import wandb

from care_nl_ica.dep_mat import jacobians
from care_nl_ica.losses.utils import Losses
from care_nl_ica.metrics.dep_mat import (
    jacobian_to_tril_and_perm,
    permutation_loss,
    check_permutation,
)
from care_nl_ica.metrics.ica_dis import (
    calc_disent_metrics,
    DisentanglementMetrics,
    frobenius_diagonality,
    corr_matrix,
)
from care_nl_ica.models.model import ContrastiveLearningModel

from care_nl_ica.metrics.dep_mat import JacobianBinnedPrecisionRecall


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
        num_thresholds: int = 10,
        log_freq=500,
    ):
        """

        :param log_freq: gradient/weight log frequency for W&B, None turns it off
        :param num_thresholds: number of thresholds for calculating the Jacobian precision-recall
        :param log_latent_rec: Log the latents and their reconstructions
        :param normalize_latents: normalize the latents to [0;1] (for the Jacobian calculation)
        :param use_bias: Use bias in the network
        :param use_flows: Use a Flow unmixing
        :param lr: learning rate
        :param latent_dim: latent dimension
        :param use_ar_mlp: Use the AR MLP unmixing
        :param device: device
        :param qr: QR loss on the bottleneck matrix
        :param triangularity_loss: triangularity loss on the correlation matrix
        :param entropy: Entropy regularizer coefficient on the Sinkhorn weights
        :param l1: L1 regularization coefficient
        :param budget: Constrain the non-zero elements on the bottleneck
        :param use_reverse: Use reverse layers in teh flow unmixing
        :param use_batch_norm: Use batchnorm layers in the Flow unmixing
        :param learnable_mask: makes the masks in the flow learnable
        :param sinkhorn: Use the Sinkhorn network
        :param triangular: Force the AR MLP bottleneck to be triangular
        :param verbose: Print out details, more logging
        :param p: Exponent of the assumed model Lp Exponential distribution
        :param tau: Print out details, more extensive logging
        :param alpha: Weight factor between the two loss components
        :param box_min: For box normalization only. Minimal value of box.
        :param box_max: For box normalization only. Maximal value of box.
        :param sphere_r: For sphere normalization only. Radius of the sphere.
        :param normalization: Output normalization to use. If empty, do not normalize at all. Can be ("", "fixed_box", "learnable_box", "fixed_sphere", "learnable_sphere")
        :param start_step: Starting step index to activate functions (e.g. the QR loss)
        """
        super().__init__()
        self.save_hyperparameters()

        self.model: ContrastiveLearningModel = ContrastiveLearningModel(
            self.hparams
        ).to(self.hparams.device)

        self.dep_mat = None
        self.qr_success: bool = False

        self._configure_metrics()

    def _configure_metrics(self):
        self.jac_prec_recall = JacobianBinnedPrecisionRecall(
            num_thresholds=self.hparams.num_thresholds
        )

    def on_train_start(self) -> None:
        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            self.logger.experiment.log({f"thresholds": self.jac_prec_recall.thresholds})

            if self.hparams.log_freq is not None:
                self.logger.watch(self.model, log="all", log_freq=self.hparams.log_freq)

    def on_epoch_end(self) -> None:
        self._set_bottleneck_with_qr_estimate()

    def _set_bottleneck_with_qr_estimate(self):
        if (
            self.qr_success is True
            and self.hparams.qr != 0.0
            and (
                self.hparams.start_step is None
                or (
                    self.hparams.start_step is not None
                    and self.global_step >= self.hparams.start_step
                )
            )
        ):
            self.hparams.qr = 0.0
            self.model.unmixing.ar_bottleneck.make_triangular_with_permute(
                self.unmixing_weight_qr_estimate, self.hard_permutation
            )

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

        self.dep_mat = self._calc_and_log_matrices(mixtures, sources)

        """Precision-Recall"""
        self.jac_prec_recall.update(
            self.dep_mat.detach(), self.trainer.datamodule.unmixing_jacobian
        )
        precisions, recalls, thresholds = self.jac_prec_recall.compute()
        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            self.logger.experiment.log(
                {
                    f"{panel_name}/jacobian/precisions": precisions,
                    f"{panel_name}/jacobian/recalls": recalls,
                }
            )

        """Disentanglement"""
        disent_metrics: DisentanglementMetrics = calc_disent_metrics(
            sources[0], reconstructions[0]
        )

        # for sweeps
        self.log("val_loss", losses.total_loss)
        self.log("val_mcc", disent_metrics.perm_score)

        self.log(f"{panel_name}/disent", disent_metrics.log_dict())

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            self.logger.experiment.log(
                {
                    f"{panel_name}/disent/non_perm_corr_mat": disent_metrics.non_perm_corr_mat
                }
            )
            self.logger.experiment.log(
                {f"{panel_name}/disent/perm_corr_mat": disent_metrics.perm_corr_mat}
            )

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

            if self.hparams.sinkhorn is True:
                self.logger.experiment.log(
                    {"sinkhorn": self.model.sinkhorn.doubly_stochastic_matrix.detach()}
                )

            if self.hparams.verbose is True:
                self.logger.experiment.log({"enc_dec_jacobian": enc_dec_jac.detach()})
                self.logger.experiment.log(
                    {"numerical_jacobian": numerical_jacobian.detach()}
                )

                # log the bottleneck weights
                if hasattr(self.model.unmixing, "ar_bottleneck") is True:
                    self.logger.experiment.log(
                        {"ar_bottleneck": self.model.unmixing.ar_bottleneck.detach()}
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

        # estimate entropy (i.e., the baseline of the loss)
        entropy_estimate, _, _ = self.model.loss(*sources, *sources)

        losses = Losses(
            cl_pos=loss_pos_mean,
            cl_neg=loss_neg_mean,
            cl_entropy=entropy_estimate,
            sinkhorn_entropy=self.model.sinkhorn_entropy_loss(),
            bottleneck_l1=self.model.bottleneck_l1_loss(),
            sparsity_budget=self.model.budget_loss(),
            triangularity=self.triangularity_loss(*sources, *reconstructions),
            qr=self.qr(),
        )

        return sources, mixtures, reconstructions, losses

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

                # inv_perm is incentivized to be the permutation for the causal ordering
                inv_perm, self.unmixing_weight_qr_estimate = jacobian_to_tril_and_perm(
                    J, self.hparams.cholesky_permutation is False
                )

                """
                The first step is to ensure that the inv_perm in the QR decomposition of the transposed(bottleneck) is 
                **a permutation** matrix.

                The second step is to ensure that the permutation matrix is the identity. If we got a permutation matrix
                in the first step, then we could use inv_perm.T to multiply the observations. 
                """

                # loss options

                if (
                    self.model.training is False
                    and isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True
                ):
                    self.logger.experiment.log({"Val/Q": inv_perm.detach()})

                if self.hparams.use_ar_mlp is True:
                    self.hard_permutation, self.qr_success = check_permutation(inv_perm)

                loss = self.hparams.qr * permutation_loss(inv_perm, matrix_power=False)

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
