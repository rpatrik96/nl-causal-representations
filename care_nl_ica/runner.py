import subprocess
import sys
from os.path import dirname

import pytorch_lightning as pl
import torch
import wandb

from care_nl_ica.dep_mat import jacobians
from care_nl_ica.independence.indep_check import IndependenceChecker
from care_nl_ica.losses.utils import ContrastiveLosses
from care_nl_ica.metrics.dep_mat import (
    JacobianBinnedPrecisionRecall,
)
from care_nl_ica.metrics.ica_dis import (
    calc_disent_metrics,
    DisentanglementMetrics,
)
from care_nl_ica.models.model import ContrastiveLearningModel


class ContrastiveICAModule(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        latent_dim: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = False,
        p: float = 1,
        tau: float = 1.0,
        alpha: float = 0.5,
        box_min: float = 0.0,
        box_max: float = 1.0,
        sphere_r: float = 1.0,
        normalization: str = "",
        start_step=None,
        use_bias=False,
        normalize_latents: bool = True,
        log_latent_rec=False,
        num_thresholds: int = 30,
        log_freq=500,
        offline: bool = False,
        num_permutations=10,
    ):
        """

        :param num_permutations: number of permutations for HSIC
        :param offline: offline W&B run (sync at the end)
        :param log_freq: gradient/weight log frequency for W&B, None turns it off
        :param num_thresholds: number of thresholds for calculating the Jacobian precision-recall
        :param log_latent_rec: Log the latents and their reconstructions
        :param normalize_latents: normalize the latents to [0;1] (for the Jacobian calculation)
        :param use_bias: Use bias in the network
        :param lr: learning rate
        :param latent_dim: latent dimension
        :param device: device
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
        self.munkres_permutation_idx = None

        self.indep_checker = IndependenceChecker(self.hparams)
        self.hsic_adj = None

        self._configure_metrics()

    def _configure_metrics(self):
        self.jac_prec_recall = JacobianBinnedPrecisionRecall(
            num_thresholds=self.hparams.num_thresholds
        )

    def on_train_start(self) -> None:
        torch.cuda.empty_cache()
        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            self.logger.experiment.log({f"thresholds": self.jac_prec_recall.thresholds})

            if self.hparams.log_freq is not None:
                self.logger.watch(self.model, log="all", log_freq=self.hparams.log_freq)

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
        self.log(
            f"{panel_name}/losses", losses.log_dict(), on_epoch=True, on_step=False
        )

        self.dep_mat = self._calc_and_log_matrices(mixtures, sources).detach()

        """Precision-Recall"""
        self.jac_prec_recall.update(
            self.dep_mat, self.trainer.datamodule.unmixing_jacobian
        )
        precisions, recalls, thresholds = self.jac_prec_recall.compute()
        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            self.logger.experiment.log(
                {
                    f"{panel_name}/jacobian/precisions": precisions,
                    f"{panel_name}/jacobian/recalls": recalls,
                }
            )
        """HSIC"""
        if (
            batch_idx == 0
            and (
                self.current_epoch % 1000 == 0
                or self.current_epoch == (self.trainer.max_epochs - 1)
            )
            is True
        ):
            self.hsic_adj = self.indep_checker.check_multivariate_dependence(
                reconstructions[0], mixtures[0]
            ).float()
            if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
                self.logger.experiment.log({f"{panel_name}/hsic_adj": self.hsic_adj})

        """Disentanglement"""
        disent_metrics, self.munkres_permutation_idx = calc_disent_metrics(
            sources[0], reconstructions[0]
        )
        self.log(
            f"{panel_name}/disent",
            disent_metrics.log_dict(),
            on_epoch=True,
            on_step=False,
        )

        # for sweeps
        self.log("val_loss", losses.total_loss, on_epoch=True, on_step=False)
        self.log("val_mcc", disent_metrics.perm_score, on_epoch=True, on_step=False)

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            self.logger.experiment.log(
                {
                    f"{panel_name}/disent/non_perm_corr_mat": disent_metrics.non_perm_corr_mat
                }
            )
            self.logger.experiment.log(
                {f"{panel_name}/disent/perm_corr_mat": disent_metrics.perm_corr_mat}
            )

        self.log_scatter_latent_rec(sources[0], reconstructions[0], "n1")
        self.log_scatter_latent_rec(mixtures[0], reconstructions[0], "z1_n1_rec")

        return losses.total_loss

    def _calc_and_log_matrices(self, mixtures, sources):
        dep_mat, numerical_jacobian, enc_dec_jac = jacobians(
            self.model, sources[0], mixtures[0]
        )

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            self.logger.experiment.summary[
                "Unmixing/unmixing_jacobian"
            ] = dep_mat.detach()

            if self.hparams.verbose is True:
                self.logger.experiment.log({"enc_dec_jacobian": enc_dec_jac.detach()})
                self.logger.experiment.log(
                    {"numerical_jacobian": numerical_jacobian.detach()}
                )

        return dep_mat

    def _forward(self, batch):
        sources, mixtures = batch
        sources, mixtures = tuple(sources), tuple(mixtures)

        # forward
        reconstructions = self.model(mixtures)
        # create random "negative" pairs
        # this is faster than sampling z3 again from the marginal distribution
        # and should also yield samples as if they were sampled from the marginal
        # z3 = torch.roll(sources[0], 1, 0)
        # z3_rec = torch.roll(reconstructions[0], 1, 0)

        _, _, [loss_pos_mean, loss_neg_mean] = self.model.loss(
            *sources,
            # z3,
            *reconstructions,
            # z3_rec
        )

        # estimate entropy (i.e., the baseline of the loss)
        entropy_estimate, _, _ = self.model.loss(
            *sources,
            # z3,
            *sources,
            # z3
        )

        losses = ContrastiveLosses(
            cl_pos=loss_pos_mean,
            cl_neg=loss_neg_mean,
            cl_entropy=entropy_estimate,
        )

        return sources, mixtures, reconstructions, losses

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
            for key, val in self.trainer.datamodule.data_to_log.items():
                self.logger.experiment.summary[key] = val

    def on_fit_end(self) -> None:
        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            """ICA permutation indices"""
            self.logger.experiment.summary[
                "munkres_permutation_idx"
            ] = self.munkres_permutation_idx

            """Jacobians"""
            table = wandb.Table(
                data=[self.dep_mat.reshape(1, -1).tolist()], columns=["dep_mat"]
            )
            self.logger.experiment.log({f"dep_mat_table": table})

            table = wandb.Table(
                data=[
                    self.trainer.datamodule.unmixing_jacobian.reshape(1, -1).tolist()
                ],
                columns=["gt_unmixing_jacobian"],
            )
            self.logger.experiment.log({f"gt_unmixing_jacobian_table": table})

            """HSIC"""
            table = wandb.Table(
                data=[self.hsic_adj.reshape(1, -1).tolist()], columns=["hsic_adj"]
            )
            self.logger.experiment.log({f"hsic_adj_table": table})

        if self.hparams.offline is True:
            # Syncing W&B at the end
            # 1. save sync dir (after marking a run finished, the W&B object changes (is teared down?)
            sync_dir = dirname(self.logger.experiment.dir)
            # 2. mark run complete
            wandb.finish()
            # 3. call the sync command for the run directory
            subprocess.check_call(["wandb", "sync", sync_dir])
