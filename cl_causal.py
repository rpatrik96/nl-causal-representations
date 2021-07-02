from collections import Counter

import numpy as np
import torch

from args import parse_args
from cl_ica import latent_spaces
from dep_mat import calc_jacobian, dependency_loss
from indep_check import IndependenceChecker
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
from prob_utils import setup_marginal, sample_marginal_and_conditional, setup_conditional, calc_disentanglement_scores, \
    check_independence_z_gz
from runner import Runner
from utils import setup_seed, save_state_dict, print_statistics, set_learning_mode, set_device

import wandb


def main():

    # setup
    args = parse_args()

    set_device(args)
    setup_seed(args.seed)
    set_learning_mode(args)

    runner = Runner(args)

    g = runner.model.decoder
    h_ind = lambda z: g(z)

    indep_checker = IndependenceChecker(args)

    wandb.init(project="test")
    wandb.config = args
    # distributions
    latent_space = latent_spaces.LatentSpace(space=(runner.model.space), sample_marginal=(setup_marginal(args)),
                                             sample_conditional=(setup_conditional(args)), )

    check_independence_z_gz(indep_checker, h_ind, latent_space)

    save_state_dict(args, g)

    for learning_mode in args.learning_modes:
        print("supervised test: {}".format(learning_mode))

        f = runner.model.encoder
        h = runner.model.h

        if ("total_loss_values" in locals() and not args.resume_training) or "total_loss_values" not in locals():
            individual_losses_values = []
            total_loss_values = []
            lin_dis_scores = []
            perm_dis_scores = []
            causal_check = []

        global_step = len(total_loss_values) + 1

        while (
                global_step <= args.n_steps if learning_mode else global_step <= (
                        args.n_steps * args.more_unsupervised)):
            data = sample_marginal_and_conditional(latent_space, size=args.batch_size, device=args.device)

            """Dependency matrix - BEGIN """

            # 1. get a sample from the latents
            # these are the noise variables in Wieland's notation
            # todo: probably we can use something from data?
            z_disentanglement = latent_space.sample_marginal(args.n_eval_samples)

            # 2. calculate the signal mixtures (i.e., the observations)
            obs = g(z_disentanglement)

            # 3. calculate the dependency matrix
            # x \times f(x)
            dep_mat = calc_jacobian(f, obs, normalize=args.preserve_vol).abs().mean(0).T
            # 4. calculate the loss for the dependency matrix
            dep_loss = dependency_loss(dep_mat)

            # todo: FISTA or similar needed
            # todo: the above part should be integrated into the training loop
            # todo: dep_loss should be added to the loss in train_and_log_losses

            """Dependency matrix - END """

            total_loss, losses = runner.train(data, h, learning_mode)

            individual_losses_values.append(losses)
            total_loss_values.append(total_loss)

            lin_dis_scores, perm_dis_scores = log_independence_and_disentanglement(args, causal_check, global_step, h,
                                                                                   h_ind, dep_mat, indep_checker,
                                                                                   latent_space, lin_dis_scores,
                                                                                   perm_dis_scores)

            print_statistics(args, causal_check, f, global_step, lin_dis_scores[-1], perm_dis_scores[-1], total_loss,
                             total_loss_values, dep_mat, dep_loss)

            wandb.log({"total_loss" : total_loss, "dep_mat" : dep_mat})

            global_step += 1

        save_state_dict(args, f, "{}_f.pth".format("sup" if learning_mode else "unsup"))
        torch.cuda.empty_cache()

        runner.reset_encoder()

    report_final_disentanglement_scores(args, h, latent_space)


def log_independence_and_disentanglement(args, causal_check, global_step, h, h_ind, dep_mat,
                                         ind_checker: IndependenceChecker, latent_space: latent_spaces.LatentSpace,
                                         lin_dis_scores: list, perm_dis_scores: list):
    if global_step % args.n_log_steps == 1 or global_step == args.n_steps:

        z_disentanglement = latent_space.sample_marginal(args.n_eval_samples)
        hz_disentanglement = h(z_disentanglement)

        lin_dis_score, perm_dis_score = calc_disentanglement_scores(z_disentanglement, hz_disentanglement)
        lin_dis_scores.append(lin_dis_score)
        perm_dis_scores.append(perm_dis_score)

        if args.use_dep_mat:
            null_list = [False] * torch.numel(dep_mat)
            null_list[torch.argmin(dep_mat).item()] = True
            var_map = [1, 1, 2, 2]
        else:
            null_list, var_map = ind_checker.check_bivariate_dependence(h_ind(z_disentanglement), hz_disentanglement)

        ######Note this is specific to a dense 2x2 triangular matrix!######
        if Counter(null_list) == Counter([False, False, False, True]):
            causal_check.append(1.)
            print('concluded a causal effect')

            for i, null in enumerate(null_list):
                if null:
                    print('cause variable is X{}'.format(str(var_map[i])))
        else:
            causal_check.append(0.)
        """
        from matplotlib import pyplot as plt
        fig_z, ax_z = plt.subplots()
        ax_z.hist(z_disentanglement.detach().cpu().numpy().flatten(), bins=100)
        writer.add_figure("hist_z", fig_z, global_step)
        fig_hz, ax_hz = plt.subplots()
        ax_hz.hist(hz_disentanglement.detach().cpu().numpy().flatten(), bins=100)
        writer.add_figure("hist_hz", fig_hz, global_step)
        writer.flush()
        """

    else:
        lin_dis_scores.append(lin_dis_scores[-1])
        perm_dis_scores.append(perm_dis_scores[-1])
        causal_check.append(causal_check[-1])
    return lin_dis_scores, perm_dis_scores


def report_final_disentanglement_scores(args, h, latent_space):
    device = args.device
    final_linear_scores = []
    final_perm_scores = []

    with torch.no_grad():
        for i in range(args.num_eval_batches):
            data = sample_marginal_and_conditional(latent_space, args.batch_size, device)
            z1, z2_con_z1, z3 = data
            z1 = z1.to(device)
            z3 = z3.to(device)
            z2_con_z1 = z2_con_z1.to(device)
            # z3 = torch.roll(z1, 1, 0)
            z1_rec = h(z1)
            z2_con_z1_rec = h(z2_con_z1)
            z3_rec = h(z3)

            linear_disentanglement_score, permutation_disentanglement_score = calc_disentanglement_scores(z1, z1_rec)
            final_linear_scores.append(linear_disentanglement_score)
            final_perm_scores.append(permutation_disentanglement_score)

    print("linear mean: {} std: {}".format(np.mean(final_linear_scores), np.std(final_linear_scores)))
    print("perm mean: {} std: {}".format(np.mean(final_perm_scores), np.std(final_perm_scores)))


if __name__ == "__main__":
    main()
