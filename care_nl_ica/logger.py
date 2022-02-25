import torch
import wandb

from .utils import matrix_to_dict


class Logger(object):
    def __init__(self, hparams, model) -> None:
        super().__init__()
        self.hparams = hparams

        self._setup_exp_management(model)

        self.total_loss_values = None

    def _setup_exp_management(self, model):
        if self.hparams.use_wandb is True:
            wandb.init(
                entity="causal-representation-learning",
                project=self.hparams.project,
                notes=self.hparams.notes,
                config=self.hparams,
                tags=self.hparams.tags,
            )
            wandb.watch(model, log_freq=self.hparams.n_log_steps, log="all")

            # define metrics
            wandb.define_metric("total_loss", summary="min")
            wandb.define_metric("lin_dis_score", summary="max")
            wandb.define_metric("perm_dis_score", summary="max")

    def log_jacobian(
        self, dep_mat, name="gt_decoder", inv_name="gt_encoder", log_inverse=True
    ):
        jac = dep_mat.detach().cpu()
        cols = [f"a_{i}" for i in range(dep_mat.shape[1])]

        gt_jacobian_dec = wandb.Table(columns=cols, data=jac.tolist())
        self.log_summary(**{f"{name}_jacobian": gt_jacobian_dec})

        if log_inverse is True:
            gt_jacobian_enc = wandb.Table(columns=cols, data=jac.inverse().tolist())
            self.log_summary(**{f"{inv_name}_jacobian": gt_jacobian_enc})
