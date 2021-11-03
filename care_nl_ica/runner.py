import torch
from torch.nn import functional as F
from torch.autograd.functional import jacobian

from dep_mat import calc_jacobian_loss, calc_jacobian, calc_jacobian_numerical
from care_nl_ica.logger import Logger
from care_nl_ica.model import ContrastiveLearningModel
from care_nl_ica.prob_utils import sample_marginal_and_conditional
from care_nl_ica.utils import unpack_item_list, save_state_dict

from metric_logger import Metrics
class Runner(object):

    def __init__(self, hparams) -> None:
        super().__init__()

        self.hparams = hparams

        self.model = ContrastiveLearningModel(self.hparams)
        self.optimizer = torch.optim.Adam(self.model.encoder.parameters(), lr=self.hparams.lr)

        self.logger = Logger(self.hparams, self.model)

        self.metrics = Metrics()

    def reset_encoder(self):
        self.model.reset_encoder()
        self.optimizer = torch.optim.Adam(self.model.encoder.parameters(), lr=self.hparams.lr)

    def train_step(self, data, h, test):
        device = self.hparams.device

        z1, z2_con_z1, z3 = data
        z1 = z1.to(device)
        z2_con_z1 = z2_con_z1.to(device)
        z3 = z3.to(device)

        # create random "negative" pairs
        # this is faster than sampling z3 again from the marginal distribution
        # and should also yield samples as if they were sampled from the marginal
        # import pdb; pdb.set_trace()
        # z3_shuffle_indices = torch.randperm(len(z1))
        # z3_shuffle_indices = torch.roll(torch.arange(len(z1)), 1)
        # z3 = z1[z3_shuffle_indices]
        # z3 = z3.to(device)

        self.optimizer.zero_grad()

        z1_rec = h(z1)
        z2_con_z1_rec = h(z2_con_z1)
        z3_rec = h(z3)
        # z3_rec = z1_rec[z3_shuffle_indices]
    

        self.logger.log_scatter_latent_rec(z1, z1_rec, "z1")
        self.logger.log_scatter_latent_rec(z2_con_z1, z2_con_z1_rec, "z2_con_z1")
        self.logger.log_scatter_latent_rec(z3, z3_rec, "z3")

        if test:
            total_loss_value = F.mse_loss(z1_rec, z1)
            losses_value = [total_loss_value]
        else:
            total_loss_value, _, losses_value = self.model.loss(
                z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec
            )

            # writer.add_scalar("loss_hz", total_loss_value, global_step)
            # writer.add_scalar("loss_z", loss(
            #    z1, z2_con_z1, z3, z1, z2_con_z1, z3
            # )[0], global_step)
            # writer.flush()

        if not self.hparams.identity_mixing_and_solution and self.hparams.lr != 0:

            if self.hparams.use_l1 is False:
                total_loss_value.backward()
            elif self.hparams.use_ar_mlp is True:
                # add sparsity loss to the AR MLP bottleneck
                (total_loss_value+self.model.hparams.l1*self.model.encoder.bottleneck_l1_norm).backward()


            self.optimizer.step()

        return total_loss_value.item(), unpack_item_list(losses_value)

    def train(self, data, h, test):
        if self.hparams.lr != 0:
            total_loss, losses = self.train_step(data, h=h, test=test)
        else:
            with torch.no_grad():
                total_loss, losses = self.train_step(data, h=h, test=test)

        return total_loss, losses

    def training_loop(self, indep_checker, latent_space):
        for learning_mode in self.hparams.learning_modes:
            print("supervised test: {}".format(learning_mode))

            self.logger.init_log_lists()

            while (self.logger.global_step <= self.hparams.n_steps if learning_mode else self.logger.global_step <= (
                    self.hparams.n_steps * self.hparams.more_unsupervised)):

                data = sample_marginal_and_conditional(latent_space, size=self.hparams.batch_size,
                                                       device=self.hparams.device)


                dep_loss, dep_mat, numerical_jacobian, enc_dec_jac = calc_jacobian_loss(self.model,latent_space)


                # Update the metrics
                # self.metrics.update(y_pred=dep_mat, y_true=gt_jacobian)

                # if self.hparams.use_flows:
                #     dep_mat = self.model.encoder.confidence.mask()

                total_loss, losses = self.train(data, self.model.h, learning_mode)

                self.logger.log(self.model.h, self.model.h_ind, dep_mat, enc_dec_jac, indep_checker, latent_space, losses,
                                total_loss, dep_loss, self.model.encoder, None, None if self.hparams.use_ar_mlp is False else self.model.encoder.ar_bottleneck.weight, numerical_jacobian)#, self.metrics.compute())

            save_state_dict(self.hparams, self.model.encoder, "{}_f.pth".format("sup" if learning_mode else "unsup"))
            torch.cuda.empty_cache()

            self.reset_encoder()

        self.logger.report_final_disentanglement_scores(self.model.h, latent_space)
