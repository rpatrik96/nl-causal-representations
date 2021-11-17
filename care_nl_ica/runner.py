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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        self.logger = Logger(self.hparams, self.model)

        self.metrics = Metrics()

    def reset_encoder(self):
        self.model.reset_encoder()
        self.optimizer = torch.optim.Adam(self.model.encoder.parameters(), lr=self.hparams.lr)

    def train_step(self, data, h, test):
        device = self.hparams.device

        n1, n2_con_n1, n3 = data
        n1 = n1.to(device)
        n2_con_n1 = n2_con_n1.to(device)
        n3 = n3.to(device)

        # create random "negative" pairs
        # this is faster than sampling n3 again from the marginal distribution
        # and should also yield samples as if they were sampled from the marginal
        # import pdb; pdb.set_trace()
        # n3_shuffle_indices = torch.randperm(len(n1))
        # n3_shuffle_indices = torch.roll(torch.arange(len(n1)), 1)
        # n3 = n1[n3_shuffle_indices]
        # n3 = n3.to(device)

        self.optimizer.zero_grad()

        n1_rec = h(n1)
        n2_con_n1_rec = h(n2_con_n1)
        n3_rec = h(n3)
        # n3_rec = n1_rec[n3_shuffle_indices]
    

        self.logger.log_scatter_latent_rec(n1, n1_rec, "n1")

        with torch.no_grad():
            z1 = self.model.decoder(n1)
            self.logger.log_scatter_latent_rec(z1, n1_rec, "z1_n1_rec")
  
        if test:
            total_loss_value = F.mse_loss(n1_rec, n1)
            losses_value = [total_loss_value]
        else:
            total_loss_value, _, losses_value = self.model.loss(
                n1, n2_con_n1, n3, n1_rec, n2_con_n1_rec, n3_rec
            )

            # writer.add_scalar("loss_hn", total_loss_value, global_step)
            # writer.add_scalar("loss_n", loss(
            #    n1, n2_con_n1, n3, n1, n2_con_n1, n3
            # )[0], global_step)
            # writer.flush()

        if not self.hparams.identity_mixing_and_solution and self.hparams.lr != 0:


            # add the learnable jacobian
            if self.hparams.learn_jacobian is True:


                # get the data after mixing (decoding)
                z1 = self.model.decoder(n1)
                z2_con_z1 = self.model.decoder(n2_con_n1)
                z3 = self.model.decoder(n3)

                # reconstruct linearly
                n1_tilde = self.model.jacob(z1)
                n2_con_n1_tilde = self.model.jacob(z2_con_z1)
                n3_tilde = self.model.jacob(z3)


                eps = 1e-8

                # standardize n
                n1_rec_std = ((n1_rec - n1_rec.mean(dim=0)) / (n1_rec.std(dim=0) + eps) + n1_rec.mean(dim=0)).detach()
                n2_con_n1_rec_std = ((n2_con_n1_rec - n2_con_n1_rec.mean(dim=0)) / (n2_con_n1_rec.std(dim=0) + eps) + n2_con_n1_rec.mean(dim=0)).detach()
                n3_rec_std = ((n3_rec - n3_rec.mean(dim=0)) / (n3_rec.std(dim=0) + eps) + n3_rec.mean(dim=0)).detach()


                lin_mse = (n1_tilde - n1_rec_std).pow(2).mean()+ (n2_con_n1_tilde - n2_con_n1_rec_std).pow(2).mean() + (n3_tilde - n3_rec_std).pow(2).mean()

                total_loss_value += lin_mse
            
            if self.hparams.l2 != 0.0:
                l2 :float= 0.0
                for param in self.model.encoder.parameters():
                    l2 += torch.sum(param**2)

                total_loss_value += self.hparams.l2 * l2

            

            
            if self.hparams.l1 != 0 and self.hparams.use_ar_mlp is True:
                # add sparsity loss to the AR MLP bottleneck
                (total_loss_value+self.hparams.l1*self.model.encoder.bottleneck_l1_norm).backward()
            else:
                total_loss_value.backward()


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
                threshold = 3e-3
                # from pdb import set_trace; set_trace()
                inv_abs_dep_mat = dep_mat.detach().inverse().abs()
                self.metrics.update(y_pred=(inv_abs_dep_mat > threshold).bool().cpu().reshape(-1,1), y_true=(self.gt_jacobian_decoder.abs()>threshold).bool().cpu().reshape(-1,1))

                # calculate the distance between ground truth and predicted jacobian
                jacobian_norm_diff =  torch.norm(inv_abs_dep_mat - self.gt_jacobian_decoder.abs())
                jacobian_thresholded_norm_diff =  torch.norm(inv_abs_dep_mat*(inv_abs_dep_mat>threshold) - self.gt_jacobian_decoder.abs())

                # if self.hparams.use_flows:
                #     dep_mat = self.model.encoder.confidence.mask()

                total_loss, losses = self.train(data, self.model.h, learning_mode)

                self.logger.log(self.model.h, self.model.h_ind, dep_mat, enc_dec_jac, indep_checker, latent_space, losses,
                                total_loss,  dep_loss,  self.model.encoder, self.metrics.compute(), None if self.hparams.use_ar_mlp is False else self.model.encoder.ar_bottleneck.weight, numerical_jacobian, None if self.hparams.learn_jacobian is False else self.model.jacob.weight, jacobian_norm_diff, jacobian_thresholded_norm_diff)

            save_state_dict(self.hparams, self.model.encoder, "{}_f.pth".format("sup" if learning_mode else "unsup"))
            torch.cuda.empty_cache()

            self.reset_encoder()

        self.logger.report_final_disentanglement_scores(self.model.h, latent_space)
