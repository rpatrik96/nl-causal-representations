import argparse
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from cl_ica import disentanglement_utils
from cl_ica import encoders
from cl_ica import invertible_network_utils
from cl_ica import latent_spaces
from cl_ica import losses
from cl_ica import spaces
from hsic import HSIC

from dep_mat import calc_dependency_matrix, dependency_loss
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = "cuda"
else:
    device = "cpu"

print("device:", device)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Disentanglement with InfoNCE/Contrastive Learning - MLP Mixing"
    )
    parser.add_argument('--num-permutations', type=int, default=50)
    parser.add_argument('--n-eval-samples', type=int, default=512)
    #############################
    parser.add_argument("--sphere-r", type=float, default=1.0)
    parser.add_argument(
        "--box-min",
        type=float,
        default=0.0,
        help="For box normalization only. Minimal value of box.",
    )
    parser.add_argument(
        "--box-max",
        type=float,
        default=1.0,
        help="For box normalization only. Maximal value of box.",
    )
    parser.add_argument("--alpha", default=0.5, type=float, help="Weight factor between the two loss components")
    parser.add_argument(
        "--normalization", choices=("", "fixed_box", "learnable_box", "fixed_sphere", "learnable_sphere"),
        help="Output normalization to use. If empty, do not normalize at all.", default=""
    )
    parser.add_argument('--mode', type=str, default='unsupervised')
    parser.add_argument(
        "--more-unsupervised",
        type=int,
        default=1,
        help="How many more steps to do for unsupervised compared to supervised training.",
    )
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument(
        "--num-eval-batches",
        type=int,
        default=10,
        help="Number of batches to average evaluation performance at the end.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--act-fct",
        type=str,
        default="leaky_relu",
        help="Activation function in mixing network g.",
    )
    parser.add_argument(
        "--c-param",
        type=float,
        default=0.05,
        help="Concentration parameter of the conditional distribution.",
    )
    parser.add_argument(
        "--m-param",
        type=float,
        default=1.0,
        help="Additional parameter for the marginal (only relevant if it is not uniform).",
    )
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument(
        "--n-mixing-layer",
        type=int,
        default=3,
        help="Number of layers in nonlinear mixing network g.",
    )
    parser.add_argument(
        "--n", type=int, default=2, help="Dimensionality of the latents."
    )
    parser.add_argument(
        "--space-type", type=str, default="box", choices=("box", "sphere", "unbounded")
    )
    parser.add_argument(
        "--m-p",
        type=int,
        default=0,
        help="Type of ground-truth marginal distribution. p=0 means uniform; "
             "all other p values correspond to (projected) Lp Exponential",
    )
    parser.add_argument(
        "--c-p",
        type=int,
        default=1,
        help="Exponent of ground-truth Lp Exponential distribution.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--p",
        type=int,
        default=1,
        help="Exponent of the assumed model Lp Exponential distribution.",
    )
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--identity-mixing-and-solution", action="store_true")

    parser.add_argument("--load-f", default=None)
    parser.add_argument("--load-g", default=None)

    parser.add_argument("--batch-size", type=int, default=6144)
    parser.add_argument("--n-log-steps", type=int, default=250)
    parser.add_argument("--n-steps", type=int, default=100001)
    parser.add_argument("--resume-training", action="store_true")
    args = parser.parse_args()

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    return args


def main():
    # setup
    args = parse_args()

    global device
    if args.no_cuda:
        device = "cpu"
        print("Using cpu")

    setup_seed(args.seed)

    space = setup_space(args)
    loss = setup_loss(args)

    ind_test = HSIC(args.num_permutations)

    # distributions
    sample_marginal = sample_from_marginal(args, device)
    sample_conditional = sample_from_conditional(args, device)

    latent_space = latent_spaces.LatentSpace(space=space, sample_marginal=sample_marginal,
                                             sample_conditional=sample_conditional, )
    g = setup_g(args, device)

    h_ind = lambda z: g(z)

    check_independence_z_gz(args, h_ind, ind_test, latent_space)

    save_state_dict(args, g)

    test_list = set_learning_mode(args)

    for test in test_list:
        print("supervised test: {}".format(test))

        f = setup_f(args, device)
        optimizer = torch.optim.Adam(f.parameters(), lr=args.lr)
        h = (lambda z: f(g(z))) if not args.identity_mixing_and_solution else (lambda z: z)

        if ("total_loss_values" in locals() and not args.resume_training) or "total_loss_values" not in locals():
            individual_losses_values = []
            total_loss_values = []
            linear_disentanglement_scores = []
            permutation_disentanglement_scores = []
            causal_check = []

        global_step = len(total_loss_values) + 1

        while (global_step <= args.n_steps if test else global_step <= (args.n_steps * args.more_unsupervised)):
            data = sample_marginal_and_conditional(latent_space, size=args.batch_size)

            """Dependency matrix - BEGIN """

            # 1. get a sample from the latents
            # these are the noise variables in Wieland's notation
            # todo: probably we can use something from data?
            z_disentanglement = latent_space.sample_marginal(args.n_eval_samples)

            # 2. calculate the signal mixtures (i.e., the observations)
            obs = g(z_disentanglement)

            # 3. calculate the dependency matrix
            dep_mat = calc_dependency_matrix(f, obs)

            # 4. calulate the loss for the dependency matrix
            dep_loss = dependency_loss(dep_mat)

            # todo: FISTA or similar needed
            # todo: the above par tshould be integrated into the training loop
            # todo: dep_loss should be added to the loss in train_and_log_losses

            """Dependency matrix - END """

            total_loss_value = train_and_log_losses(args, data, individual_losses_values, loss, optimizer,
                                                    total_loss_values, h, test)

            linear_disentanglement_score, permutation_disentanglement_score \
                = log_independence_and_disentanglement(args,
                                                       causal_check,
                                                       global_step,
                                                       h,
                                                       h_ind,
                                                       ind_test,
                                                       latent_space,
                                                       linear_disentanglement_scores,
                                                       permutation_disentanglement_scores)

            print_statistics(args, causal_check, f, global_step, linear_disentanglement_score,
                             permutation_disentanglement_score, total_loss_value, total_loss_values)

            global_step += 1

        save_state_dict(args, f, "{}_f.pth".format("sup" if test else "unsup"))
        torch.cuda.empty_cache()

    report_final_disentanglement_scores(args, device, h, latent_space)


def train_step(args, data, loss, optimizer, h, test):
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

    optimizer.zero_grad()

    z1_rec = h(z1)
    z2_con_z1_rec = h(z2_con_z1)
    z3_rec = h(z3)
    # z3_rec = z1_rec[z3_shuffle_indices]

    if test:
        total_loss_value = F.mse_loss(z1_rec, z1)
        losses_value = [total_loss_value]
    else:
        total_loss_value, _, losses_value = loss(
            z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec
        )

        # writer.add_scalar("loss_hz", total_loss_value, global_step)
        # writer.add_scalar("loss_z", loss(
        #    z1, z2_con_z1, z3, z1, z2_con_z1, z3
        # )[0], global_step)
        # writer.flush()

    if not args.identity_mixing_and_solution and args.lr != 0:
        total_loss_value.backward()
        optimizer.step()

    return total_loss_value.item(), unpack_item_list(losses_value)


def train_and_log_losses(args, data, individual_losses_values, loss, optimizer, total_loss_values, h, test):
    if args.lr != 0:
        total_loss_value, losses_value = train_step(args, data, loss=loss, optimizer=optimizer, h=h, test=test)
    else:
        with torch.no_grad():
            total_loss_value, losses_value = train_step(args, data, loss=loss, optimizer=optimizer, h=h, test=test)
    total_loss_values.append(total_loss_value)
    individual_losses_values.append(losses_value)
    return total_loss_value


def log_independence_and_disentanglement(args, causal_check, global_step, h, h_ind, ind_test, latent_space,
                                         linear_disentanglement_scores, permutation_disentanglement_scores):
    if global_step % args.n_log_steps == 1 or global_step == args.n_steps:

        z_disentanglement = latent_space.sample_marginal(args.n_eval_samples)
        hz_disentanglement = h(z_disentanglement)

        linear_disentanglement_score, permutation_disentanglement_score = calc_disentanglement_scores(
            z_disentanglement, hz_disentanglement)
        null_list, var_map = check_bivariate_dependence(ind_test, h_ind(z_disentanglement), hz_disentanglement)

        permutation_disentanglement_scores.append(permutation_disentanglement_score)

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
        linear_disentanglement_scores.append(linear_disentanglement_scores[-1])
        permutation_disentanglement_scores.append(permutation_disentanglement_scores[-1])
        causal_check.append(causal_check[-1])
    return linear_disentanglement_score, permutation_disentanglement_score


def report_final_disentanglement_scores(args, device, h, latent_space):
    final_linear_scores = []
    final_perm_scores = []

    with torch.no_grad():
        for i in range(args.num_eval_batches):
            data = sample_marginal_and_conditional(latent_space, args.batch_size)
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


def print_statistics(args, causal_check, f, global_step, linear_disentanglement_score,
                     permutation_disentanglement_score, total_loss_value, total_loss_values):
    if global_step % args.n_log_steps == 1 or global_step == args.n_steps:
        print(
            f"Step: {global_step} \t",
            f"Loss: {total_loss_value:.4f} \t",
            f"<Loss>: {np.mean(np.array(total_loss_values[-args.n_log_steps:])):.4f} \t",
            f"Lin. Disentanglement: {linear_disentanglement_score:.4f} \t",
            f"Perm. Disentanglement: {permutation_disentanglement_score:.4f}",
            f"Causal. Check: {causal_check[-1]:.4f}",
        )
        if args.normalization == "learnable_sphere":
            print(f"r: {f[-1].r}")


def set_learning_mode(args):
    if args.mode == 'unsupervised':
        test_list = [False]
    elif args.mode == 'supervised':
        test_list = [True]
    else:
        test_list = [True, False]
    return test_list


def save_state_dict(args, model, pth="g.pth"):
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model.state_dict(), os.path.join(args.save_dir, pth))


def check_independence_z_gz(args, h_ind, ind_test, latent_space):
    z_disentanglement = latent_space.sample_marginal(args.n_eval_samples)
    linear_disentanglement_score, permutation_disentanglement_score = calc_disentanglement_scores(z_disentanglement,
                                                                                                  h_ind(
                                                                                                      z_disentanglement))
    null_list, var_map = check_bivariate_dependence(ind_test, h_ind(z_disentanglement), z_disentanglement)
    print(f"Id. Lin. Disentanglement: {linear_disentanglement_score:.4f}")
    print(f"Id. Perm. Disentanglement: {permutation_disentanglement_score:.4f}")
    print('Run test with ground truth sources')
    if Counter(null_list) == Counter([False, False, False, True]):

        print('concluded a causal effect')

        for i, null in enumerate(null_list):
            if null:
                print('cause variable is X{}'.format(str(var_map[i])))

    else:
        print('no causal effect...?')
        sys.exit()


def calc_disentanglement_scores(z, hz):
    (linear_disentanglement_score, _), _ = disentanglement_utils.linear_disentanglement(z, hz, mode="r2")
    (permutation_disentanglement_score, _,), _ = disentanglement_utils.permutation_disentanglement(
        z,
        hz,
        mode="pearson",
        solver="munkres",
        rescaling=True,
    )

    return linear_disentanglement_score, permutation_disentanglement_score


def setup_f(args, device):
    output_normalization, output_normalization_kwargs = configure_output_normalization(args)

    f = encoders.get_mlp(
        n_in=args.n,
        n_out=args.n,
        layers=[
            args.n * 10,
            args.n * 50,
            args.n * 50,
            args.n * 50,
            args.n * 50,
            args.n * 10,
        ],
        output_normalization=output_normalization,
        output_normalization_kwargs=output_normalization_kwargs
    )
    f = f.to(device)
    if args.load_f is not None:
        f.load_state_dict(torch.load(args.load_f, map_location=device))
    print("f: ", f)
    return f


def configure_output_normalization(args):
    output_normalization = None
    output_normalization_kwargs = None
    if args.normalization == "learnable_box":
        output_normalization = "learnable_box"
    elif args.normalization == "fixed_box":
        output_normalization = "fixed_box"
        output_normalization_kwargs = dict(init_abs_bound=args.box_max - args.box_min)
    elif args.normalization == "learnable_sphere":
        output_normalization = "learnable_sphere"
    elif args.normalization == "fixed_sphere":
        output_normalization = "fixed_sphere"
        output_normalization_kwargs = dict(init_r=args.sphere_r)
    elif args.normalization == "":
        print("Using no output normalization")
        output_normalization = None
    else:
        raise ValueError("Invalid output normalization:", args.normalization)
    return output_normalization, output_normalization_kwargs


def setup_g(args, device):
    # create MLP
    g = invertible_network_utils.construct_invertible_mlp(
        n=args.n,
        n_layers=args.n_mixing_layer,
        act_fct=args.act_fct,
        cond_thresh_ratio=0.001,
        n_iter_cond_thresh=25000,
        lower_triangular=True,
        weight_matrix_init='rvs'
    )

    # allocate to device
    g = g.to(device)

    # load if needed
    if args.load_g is not None:
        g.load_state_dict(torch.load(args.load_g, map_location=device))

    # make it non-trainable
    for p in g.parameters():
        p.requires_grad = False

    return g


def sample_from_conditional(args, device):
    if args.c_p == 1:
        sample_conditional = lambda space, z, size, device=device: space.laplace(
            z, args.c_param, size, device
        )
    elif args.c_p == 2:
        sample_conditional = lambda space, z, size, device=device: space.normal(
            z, args.c_param, size, device
        )
    elif args.c_p == 0:
        sample_conditional = (
            lambda space, z, size, device=device: space.von_mises_fisher(
                z, args.c_param, size, device,
            )
        )
    else:
        sample_conditional = (
            lambda space, z, size, device=device: space.generalized_normal(
                z, args.c_param, p=args.c_p, size=size, device=device
            )
        )
    return sample_conditional


def sample_from_marginal(args, device):
    eta = torch.zeros(args.n)
    if args.space_type == "sphere":
        eta[0] = args.sphere_r

    if args.m_p == 1:
        sample_marginal = lambda space, size, device=device: space.laplace(
            eta, args.m_param, size, device
        )
    elif args.m_p == 2:
        sample_marginal = lambda space, size, device=device: space.normal(
            eta, args.m_param, size, device
        )
    elif args.m_p == 0:
        sample_marginal = lambda space, size, device=device: space.uniform(
            size, device=device
        )
    else:
        sample_marginal = (
            lambda space, size, device=device: space.generalized_normal(
                eta, args.m_param, p=args.m_p, size=size, device=device
            )
        )
    return sample_marginal


def sample_marginal_and_conditional(latent_space, size, device=device):
    z = latent_space.sample_marginal(size=size, device=device)
    z3 = latent_space.sample_marginal(size=size, device=device)
    z_tilde = latent_space.sample_conditional(z, size=size, device=device)

    return z, z_tilde, z3


def setup_loss(args):
    if args.p:
        """
        loss = losses.LpSimCLRLoss(
            p=args.p, tau=args.tau, simclr_compatibility_mode=False, alpha=args.alpha, simclr_denominator=True
        )
        """
        """
        loss = losses.LpSimCLRLoss(
            p=args.p, tau=args.tau, simclr_compatibility_mode=True, alpha=args.alpha, simclr_denominator=False
        )
        """
        loss = losses.LpSimCLRLoss(
            p=args.p, tau=args.tau, simclr_compatibility_mode=True
        )
    else:
        loss = losses.SimCLRLoss(normalize=False, tau=args.tau, alpha=args.alpha)
    return loss


def setup_space(args):
    if args.space_type == "box":
        space = spaces.NBoxSpace(args.n, args.box_min, args.box_max)
    elif args.space_type == "sphere":
        space = spaces.NSphereSpace(args.n, args.sphere_r)
    else:
        space = spaces.NRealSpace(args.n)
    return space


def setup_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)


def check_bivariate_dependence(ind_test: HSIC, x1, x2):
    decisions = []
    var_map = [1, 1, 2, 2]
    with torch.no_grad():
        decisions.append(ind_test.run_test(x1[:, 0], x2[:, 1], device="cpu", bonferroni=4).item())
        decisions.append(ind_test.run_test(x1[:, 0], x2[:, 0], device="cpu", bonferroni=4).item())
        decisions.append(ind_test.run_test(x1[:, 1], x2[:, 0], device="cpu", bonferroni=4).item())
        decisions.append(ind_test.run_test(x1[:, 1], x2[:, 1], device="cpu", bonferroni=4).item())

    return decisions, var_map


def unpack_item_list(lst):
    if isinstance(lst, tuple):
        lst = list(lst)
    result_list = []
    for it in lst:
        if isinstance(it, (tuple, list)):
            result_list.append(unpack_item_list(it))
        else:
            result_list.append(it.item())
    return result_list


if __name__ == "__main__":
    main()
