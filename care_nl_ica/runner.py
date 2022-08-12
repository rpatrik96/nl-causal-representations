import subprocess
from os.path import dirname

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchmetrics import Accuracy

from IIA.igcl import igcl
from IIA.itcl import itcl
from care_nl_ica.dep_mat import calc_jacobian
from care_nl_ica.dep_mat import jacobians
from care_nl_ica.losses.utils import ContrastiveLosses
from care_nl_ica.metrics.dep_mat import JacobianBinnedPrecisionRecall
from care_nl_ica.metrics.ica_dis import (
    calc_disent_metrics,
    DisentanglementMetrics,
)
from care_nl_ica.models.model import ContrastiveLearningModel


class ModuleBase(pl.LightningModule):
    def _calc_and_log_matrices(self, sources, inputs):

        raise NotImplementedError

    def on_fit_end(self) -> None:

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
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

        if self.hparams.offline is True:
            # Syncing W&B at the end
            # 1. save sync dir (after marking a run finished, the W&B object changes (is teared down?)
            sync_dir = dirname(self.logger.experiment.dir)
            # 2. mark run complete
            wandb.finish()
            # 3. call the sync command for the run directory
            subprocess.check_call(["wandb", "sync", sync_dir])


class IIAModule(ModuleBase):
    def __init__(
        self,
        num_data=2**18,  # number of data points
        num_layer=3,  # number of layers of mixing-MLP
        num_comp=20,  # number of components (dimension)
        num_basis=64,  # number of frequencies of fourier bases
        ar_order=1,
        initial_learning_rate=0.1,  # initial learning rate (default:0.1)
        momentum=0.9,  # momentum parameter of SGD
        max_steps=int(3e6),  # number of iterations (mini-batches)
        decay_steps=int(1e6),  # decay steps (tf.train.exponential_decay)
        decay_factor=0.1,  # decay factor (tf.train.exponential_decay)
        batch_size=512,  # mini-batch size
        moving_average_decay=0.999,  # moving average decay of variables to be saved
        checkpoint_steps=int(1e7),  # interval to save checkpoint
        summary_steps=int(1e4),  # interval to save summary
        apply_pca=True,  # apply PCA for preprocessing or not
        weight_decay=1e-5,  # weight decay
        net_model="itcl",
        num_segment=256,  # learn by IIA-TCL should be None for IGCL
        normalize_latents=True,
        offline: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # MLP ---------------------------------------------------------
        list_hidden_nodes = [4 * self.hparams.num_comp] * (
            self.hparams.num_layer - 1
        ) + [self.hparams.num_comp]
        list_hidden_nodes_z = None
        # list of the number of nodes of each hidden layer of feature-MLP
        # [layer1, layer2, ..., layer(num_layer)]

        # define network
        if self.hparams.net_model == "itcl":
            self.model = itcl.Net(
                h_sizes=list_hidden_nodes,
                h_sizes_z=list_hidden_nodes_z,
                ar_order=self.hparams.ar_order,
                num_dim=self.hparams.num_comp,
                num_class=self.hparams.num_segment,
            )
            self.loss = nn.CrossEntropyLoss()

        elif self.hparams.net_model == "igcl":
            self.model = igcl.NetGaussScaleMean(
                h_sizes=list_hidden_nodes,
                h_sizes_z=list_hidden_nodes_z,
                ar_order=self.hparams.ar_order,
                num_dim=self.hparams.num_comp,
                num_data=self.hparams.num_data,
                num_basis=self.hparams.num_basis,
            )
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError

        self.configure_metrics()

        self.model.hparams = self.hparams

    def configure_metrics(self):
        self.accuracy = Accuracy()

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.hparams.initial_learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.decay_steps,
            gamma=self.hparams.decay_factor,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch):
        obs, labels, sources, true_logits = batch
        loss, _ = self._forward(labels, obs, true_logits)

        panel_name = "Train"
        self.log(f"{panel_name}/loss", loss, on_epoch=True, on_step=False)

        return loss

    def _forward(self, labels, obs, true_logits):
        """

        :param labels:
        :param obs:
        :param true_logits:
        :return: (loss, reconstructions)
        """
        if self.hparams.net_model == "itcl":
            loss, reconstructions = self._itcl_training_step(obs, labels)
        elif self.hparams.net_model == "igcl":
            # need to overwrite as different datasets have different lengths
            self.model.num_data = len(
                self.trainer.datamodule.train_dataloader().dataset
            )
            loss, reconstructions = self._igcl_training_step(obs, labels, true_logits)

            # this stacks `h` 2x, we only need it once
            reconstructions = reconstructions[: reconstructions.shape[0] // 2, :]

        return loss, reconstructions

    def validation_step(self, batch, batch_idx):
        obs, labels, sources, true_logits = batch
        loss, reconstructions = self._forward(labels, obs, true_logits)

        if self.hparams.net_model == "itcl":
            self.dep_mat = self._calc_and_log_matrices(obs.squeeze()).detach()
        elif self.hparams.net_model == "igcl":
            self.dep_mat = self._calc_and_log_matrices(
                inputs=obs.squeeze(), aux_inputs=labels.float().squeeze()
            ).detach()

        panel_name = "Val"
        self.log(f"{panel_name}/loss", loss, on_epoch=True, on_step=False)

        """Disentanglement"""
        disent_metrics: DisentanglementMetrics = calc_disent_metrics(
            sources.squeeze(), reconstructions
        )

        self.log(
            f"{panel_name}/disent",
            disent_metrics.log_dict(),
            on_epoch=True,
            on_step=False,
        )

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            self.logger.experiment.log(
                {
                    f"{panel_name}/disent/non_perm_corr_mat": disent_metrics.non_perm_corr_mat
                }
            )
            self.logger.experiment.log(
                {f"{panel_name}/disent/perm_corr_mat": disent_metrics.perm_corr_mat}
            )

        return loss

    def _calc_and_log_matrices(self, inputs, aux_inputs=None):

        """
        For ITCL and IGCL, the model returns a tuple, where index=1 equals the inverse model
        However, this inverse model ONLY includes the estimate of s_t (cf Eq 3 of the original paper), since
        x_{t-1} does not need to be estimated.

        Passing the inverse model to `calc_jacobian` will yield a matrix of [batch, dim, ts, dim], where ts equals
        the order of the NVAR model +1 (i.e., for the standard NVAR(1) model in the paper, ts=2 = p+1,
        where p is the order of the process)

        :param aux_inputs: the vector u (time) for IGCL
        :param inputs:
        :return: mean jacobian (calculated across the batch) with dimensions [dim, ts, dim], where
        [dim, 0, dim] contains the Jacobian of s_t w.r.t of x_t
        [dim, 1, dim] contains the Jacobian of s_t w.r.t of x_{t-1}
        """

        dep_mat = (
            calc_jacobian(
                self.model, inputs, normalize=False, output_idx=1, aux_inputs=aux_inputs
            )
            .abs()
            .mean(0)
            .detach()
        )

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            self.logger.experiment.summary[
                "Unmixing/unmixing_jacobian"
            ] = dep_mat.detach()

        return dep_mat

    def training_epoch_end(self, outputs) -> None:
        self.log("train_acc_epoch", self.accuracy)

    def _itcl_training_step(self, obs, labels):
        """

        :param obs:
        :param labels:
        :return: (loss, reconstructions)
        """
        logits, h, hz = self.model(obs.squeeze())

        self.accuracy(logits, labels.squeeze())

        return self.loss(logits, labels.squeeze()), h

    def _igcl_training_step(self, obs, time_indices, true_logits):
        """

        :param obs:
        :param time_indices:
        :param true_logits:
        :return: (loss, reconstructions)
        """

        logits, h, hz, _, _ = self.model(obs.squeeze(), time_indices.squeeze())
        self.accuracy(logits, true_logits.squeeze().long())

        # constraint
        self.model.a.weight.data = self.model.a.weight.data.clamp(min=0)
        self.model.b.weight.data = self.model.b.weight.data.clamp(min=0)
        self.model.c.weight.data = self.model.c.weight.data.clamp(min=0)
        self.model.d.weight.data = self.model.d.weight.data.clamp(min=0)
        self.model.e.weight.data = self.model.e.weight.data.clamp(min=0)
        self.model.f.weight.data = self.model.f.weight.data.clamp(min=0)
        self.model.g.weight.data = self.model.g.weight.data.clamp(min=0)

        return self.loss(logits, true_logits.squeeze()), h


class ContrastiveICAModule(ModuleBase):
    def __init__(
        self,
        lr: float = 1e-4,
        latent_dim: int = 3,
        use_ar_mlp: bool = True,
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
    ):
        """

        :param offline: offline W&B run (sync at the end)
        :param log_freq: gradient/weight log frequency for W&B, None turns it off
        :param num_thresholds: number of thresholds for calculating the Jacobian precision-recall
        :param log_latent_rec: Log the latents and their reconstructions
        :param normalize_latents: normalize the latents to [0;1] (for the Jacobian calculation)
        :param use_bias: Use bias in the network
        :param lr: learning rate
        :param latent_dim: latent dimension
        :param use_ar_mlp: Use the AR MLP unmixing
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

        self.dep_mat = self._calc_and_log_matrices((mixtures[0], sources[0])).detach()

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

        """Disentanglement"""
        disent_metrics: DisentanglementMetrics = calc_disent_metrics(
            sources[0], reconstructions[0]
        )

        # for sweeps
        self.log("val_loss", losses.total_loss, on_epoch=True, on_step=False)
        self.log("val_mcc", disent_metrics.perm_score, on_epoch=True, on_step=False)

        self.log(
            f"{panel_name}/disent",
            disent_metrics.log_dict(),
            on_epoch=True,
            on_step=False,
        )

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
        _, _, [loss_pos_mean, loss_neg_mean] = self.model.loss(
            *sources, *reconstructions
        )

        # estimate entropy (i.e., the baseline of the loss)
        entropy_estimate, _, _ = self.model.loss(*sources, *sources)

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
