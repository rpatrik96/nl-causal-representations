import itertools

import torch
from torch.nn import functional as F

from care_nl_ica.logger import Logger
from care_nl_ica.metric_logger import JacobianMetrics
from care_nl_ica.model import ContrastiveLearningModel
from care_nl_ica.prob_utils import sample_marginal_and_conditional, amari_distance
from care_nl_ica.utils import unpack_item_list, save_state_dict
from cl_ica import latent_spaces
from dep_mat import calc_jacobian_loss
from indep_check import IndependenceChecker
from metric_logger import Metrics
from prob_utils import setup_marginal, setup_conditional, frobenius_diagonality


class Runner(object):

    def __init__(self, hparams) -> None:
        super().__init__()

        self.hparams = hparams

        self.indep_checker = IndependenceChecker(self.hparams)
        self.model = ContrastiveLearningModel(self.hparams)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        self.logger = Logger(self.hparams, self.model)
        self.metrics = Metrics()
        self.latent_space = latent_spaces.LatentSpace(space=(self.model.space),
                                                      sample_marginal=(setup_marginal(self.hparams)),
                                                      sample_conditional=(setup_conditional(self.hparams)), )

        self._calc_dep_mat()
        self._inject_encoder_structure()

        self.dep_loss = None

        self._calc_possible_causal_orderings()

    def _calc_dep_mat(self) -> None:
        dep_mat = self.indep_checker.check_independence_z_gz(self.model.decoder, self.latent_space)
        # save the ground truth jacobian of the decoder
        if dep_mat is not None:

            # save the decoder jacobian including the permutation
            self.gt_jacobian_decoder_permuted = dep_mat.detach()
            if self.hparams.permute is True:
                # print(f"{dep_mat=}")
                # set_trace()
                dep_mat = dep_mat[torch.argsort(self.model.decoder.permute_indices), :]

            self.gt_jacobian_decoder = dep_mat.detach()
            self.gt_jacobian_encoder = torch.tril(dep_mat.detach().inverse())

            print(f"{self.gt_jacobian_encoder=}")

            self.logger.log_jacobian(dep_mat)

            self._calc_indirect_causes()

            if self.hparams.permute is True:
                self.logger.log_summary(**{"permute_indices": self.model.decoder.permute_indices})

    def _calc_possible_causal_orderings(self):
        """
        The function calculates the possible causal orderings based on the adjacency matrix

        :return:
        """
        dim = self.gt_jacobian_encoder.shape[0]

        # get all indices for nonzero elements
        nonzero_indices = (self.gt_jacobian_encoder.abs() > 0).nonzero()

        smallest_idx = []
        biggest_idx = []
        idx_range = []

        for i in range(dim):
            # select nonzero indices for the current row
            # and take the column index of the first element
            # this gives the smallest index of variable "i" in the causal ordering
            nonzero_in_row = nonzero_indices[nonzero_indices[:, 0] == i, :]

            smallest_idx.append(0 if (tmp := nonzero_in_row[0][1]) == i else smallest_idx[tmp] + 1)

            # select nonzero indices for the current columns
            # and take the row index of the first element
            # this gives the biggest index of variable "i" in the causal ordering
            nonzero_in_col = nonzero_indices[nonzero_indices[:, 1] == i, :]

            biggest_idx.append(nonzero_in_col[0][0])

            # this means that there is only 1 appearance of variable i,
            # so it can be everywhere in the causal ordering
            if len(nonzero_in_row) == 1 and len(nonzero_in_col) == 1 and smallest_idx[i] == i and biggest_idx[i] == i:

                idx_range.append(list(range(dim)))
            else:

                idx_range.append(list(range(smallest_idx[i], biggest_idx[i] + 1)))

        self.orderings = [x for x in list(itertools.product(*idx_range)) if len(set(x)) == dim]

        print(f'{self.orderings=}')

        if self.hparams.use_wandb is True:
            self.logger.log_summary(**{"causal_orderings": self.orderings})

    def _calc_indirect_causes(self) -> None:
        """
        Calculates all indirect paths in the encoder (SEM/SCM)
        :return:
        """

        # calculate the indirect cause mask
        eps = 1e-6
        matrix_power = direct_causes = torch.tril((self.gt_jacobian_encoder.abs() > eps).float(), -1)
        indirect_causes = torch.zeros_like(self.gt_jacobian_encoder)

        # add together the matrix powers of the adjacency matrix
        # this yields all indirect paths
        for i in range(self.gt_jacobian_encoder.shape[0]):
            matrix_power = matrix_power @ direct_causes

            if matrix_power.sum() == 0:
                break

            indirect_causes += matrix_power

        self.indirect_causes = indirect_causes.bool().float()  # convert all non-1 value to 1 (for safety)
        # correct for causes where both the direct and indirect paths are present
        self.indirect_causes = self.indirect_causes * ((self.indirect_causes - direct_causes) > 0).float()

        print(f"{self.indirect_causes=}")

    def _inject_encoder_structure(self) -> None:
        if self.hparams.inject_structure is True:
            if self.hparams.use_flows:
                self.model.encoder.confidence.inject_structure(self.gt_jacobian_encoder, self.hparams.inject_structure)

            elif self.hparams.use_ar_mlp:
                self.model.encoder.ar_bottleneck.inject_structure(self.gt_jacobian_encoder,
                                                                  self.hparams.inject_structure)

    def reset_encoder(self) -> None:
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
                n2_con_n1_rec_std = ((n2_con_n1_rec - n2_con_n1_rec.mean(dim=0)) / (
                        n2_con_n1_rec.std(dim=0) + eps) + n2_con_n1_rec.mean(dim=0)).detach()
                n3_rec_std = ((n3_rec - n3_rec.mean(dim=0)) / (n3_rec.std(dim=0) + eps) + n3_rec.mean(dim=0)).detach()

                lin_mse = (n1_tilde - n1_rec_std).pow(2).mean() + (n2_con_n1_tilde - n2_con_n1_rec_std).pow(
                    2).mean() + (n3_tilde - n3_rec_std).pow(2).mean()

                total_loss_value += lin_mse

            if self.hparams.l2 != 0.0:
                l2: float = 0.0
                for param in self.model.encoder.parameters():
                    l2 += torch.sum(param ** 2)

                total_loss_value += self.hparams.l2 * l2

            if self.hparams.l1 != 0 and self.hparams.use_ar_mlp is True:
                # add sparsity loss to the AR MLP bottleneck
                total_loss_value += self.hparams.l1 * self.model.encoder.bottleneck_l1_norm

            if self.hparams.permute is True:
                probs = torch.nn.functional.softmax(self.model.encoder.sinkhorn.doubly_stochastic_matrix.data, -1).view(
                    -1, )

                total_loss_value += self.hparams.entropy_coeff * torch.distributions.Categorical(probs).entropy()

            if self.dep_loss is not None:
                total_loss_value += self.dep_loss

            if self.hparams.triangularity_loss != 0.:
                from prob_utils import corr_matrix
                from dep_mat import triangularity_loss

                # todo: these use the ground truth
                # still, they can be used to show that some supervision helps
                # pearson_n1 = corr_matrix(n1.T, n1_rec.T)
                # pearson_n2_con_n1 = corr_matrix(n2_con_n1.T, n2_con_n1_rec.T)
                # pearson_n3 = corr_matrix(n3.T, n3_rec.T)
                # total_loss_value += self.hparams.diagonality_loss*(frobenius_diagonality(pearson_n1.abs()) + frobenius_diagonality(
                #     pearson_n2_con_n1.abs()) + frobenius_diagonality(pearson_n3.abs()))

                # correlation between observation and reconstructed latents
                # exploits the assumption that the SEM has a lower-triangular Jacobian
                # order is important due to the triangularity loss
                pearson_n1 = corr_matrix( self.model.decoder(n1).T, n1_rec.T)
                pearson_n2_con_n1 = corr_matrix(self.model.decoder(n2_con_n1).T, n2_con_n1_rec.T)
                pearson_n3 = corr_matrix(self.model.decoder(n3).T, n3_rec.T)
                total_loss_value += self.hparams.triangularity_loss * (
                            triangularity_loss(pearson_n1) + triangularity_loss(
                        pearson_n2_con_n1) + triangularity_loss(pearson_n3))

                # cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                # pearson_n1 = cos_sim(n1 - n1.mean(dim=1, keepdim=True), n1_rec - n1_rec.mean(dim=1, keepdim=True))
                # pearson_n2_con_n1 = cos_sim(n2_con_n1 - n2_con_n1.mean(dim=1, keepdim=True), n2_con_n1_rec - n2_con_n1_rec.mean(dim=1, keepdim=True))
                # pearson_n3 = cos_sim(n3 - n3.mean(dim=1, keepdim=True), n3_rec - n3_rec.mean(dim=1, keepdim=True))

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

    def training_loop(self):
        for learning_mode in self.hparams.learning_modes:
            print("supervised test: {}".format(learning_mode))

            self.logger.init_log_lists()

            while (self.logger.global_step <= self.hparams.n_steps if learning_mode else self.logger.global_step <= (
                    self.hparams.n_steps * self.hparams.more_unsupervised)):
                data = sample_marginal_and_conditional(self.latent_space, size=self.hparams.batch_size,
                                                       device=self.hparams.device)

                dep_loss, dep_mat, numerical_jacobian, enc_dec_jac = calc_jacobian_loss(self.model, self.latent_space)

                self.dep_loss = dep_loss

                # Update the metrics
                threshold = 3e-5
                dep_mat = dep_mat.detach()
                self.metrics.update(y_pred=(dep_mat.abs() > threshold).bool().cpu().reshape(-1, 1),
                                    y_true=(self.gt_jacobian_encoder.abs() > threshold).bool().cpu().reshape(-1, 1))

                jacobian_metrics = self._dep_mat_metrics(dep_mat, threshold)

                # if self.hparams.use_flows:
                #     dep_mat = self.model.encoder.confidence.mask()

                total_loss, losses = self.train(data, self.model.h, learning_mode)

                self.logger.log(self.model.h, self.model.h_ind, dep_mat, enc_dec_jac, self.indep_checker,
                                self.latent_space, losses, total_loss, dep_loss, self.model.encoder,
                                self.metrics.compute(),
                                None if self.hparams.use_ar_mlp is False else self.model.encoder.ar_bottleneck.assembled_weight,
                                numerical_jacobian,
                                None if self.hparams.learn_jacobian is False else self.model.jacob.weight,
                                jacobian_metrics, None if (
                            self.hparams.permute is False or self.hparams.use_sem is False) else self.model.encoder.sinkhorn.doubly_stochastic_matrix)

            save_state_dict(self.hparams, self.model.encoder, "{}_f.pth".format("sup" if learning_mode else "unsup"))
            torch.cuda.empty_cache()

            self.reset_encoder()

        self.logger.log_jacobian(dep_mat, "learned_last", log_inverse=False)
        self.logger.report_final_disentanglement_scores(self.model.h, self.latent_space)

    def _dep_mat_metrics(self, dep_mat: torch.Tensor, threshold: float = 1e-3) -> JacobianMetrics:
        # calculate the optimal threshold for 1 accuracy
        # calculate the indices where the GT is 0 (in the lower triangular part)
        sparsity_mask = (torch.tril(self.gt_jacobian_encoder.abs() < 1e-6)).bool()

        if sparsity_mask.sum() > 0:
            optimal_threshold = dep_mat[sparsity_mask].abs().max()
        else:
            optimal_threshold = None

        # calculate the distance between ground truth and predicted jacobian
        norm_diff: float = torch.norm(dep_mat.abs() - self.gt_jacobian_encoder.abs()).mean()
        thresholded_norm_diff: float = torch.norm(
            dep_mat.abs() * (dep_mat.abs() > threshold) - self.gt_jacobian_encoder.abs()).mean()

        # calculate the fraction of correctly identified zeroes
        incorrect_edges: float = ((dep_mat.abs() * self.indirect_causes) > threshold).sum()
        sparsity_accuracy: float = 1. - incorrect_edges / (self.indirect_causes.sum() + 1e-8)

        metrics = JacobianMetrics(norm_diff, thresholded_norm_diff, optimal_threshold, sparsity_accuracy,
                                  amari_distance(dep_mat, self.gt_jacobian_decoder_permuted))

        return metrics
