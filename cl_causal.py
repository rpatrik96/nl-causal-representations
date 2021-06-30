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

from args import parse_args

from dep_mat import calc_jacobian, dependency_loss
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
from prob_utils import sample_from_marginal, sample_marginal_and_conditional, sample_from_conditional
from utils import unpack_item_list, setup_seed, save_state_dict, print_statistics, set_learning_mode, set_device


def main():
    # setup
    args = parse_args()


    device = set_device(args)

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
            data = sample_marginal_and_conditional(latent_space, size=args.batch_size,device=device)

            """Dependency matrix - BEGIN """

            # 1. get a sample from the latents
            # these are the noise variables in Wieland's notation
            # todo: probably we can use something from data?
            z_disentanglement = latent_space.sample_marginal(args.n_eval_samples)

            # 2. calculate the signal mixtures (i.e., the observations)
            obs = g(z_disentanglement)

            # 3. calculate the dependency matrix
            #x \times f(x)
            dep_mat = calc_jacobian(f, obs, normalize=args.preserve_vol).abs().mean(0).T
            # 4. calculate the loss for the dependency matrix
            dep_loss = dependency_loss(dep_mat)

            # todo: FISTA or similar needed
            # todo: the above part should be integrated into the training loop
            # todo: dep_loss should be added to the loss in train_and_log_losses

            """Dependency matrix - END """

            total_loss_value = train_and_log_losses(args, data, individual_losses_values, loss, optimizer,
                                                    total_loss_values, h, test, device)

            linear_disentanglement_scores, permutation_disentanglement_scores \
                = log_independence_and_disentanglement(args,
                                                       causal_check,
                                                       global_step,
                                                       h,
                                                       h_ind,
                                                       dep_mat,
                                                       ind_test,
                                                       latent_space,
                                                       linear_disentanglement_scores,
                                                       permutation_disentanglement_scores)

            print_statistics(args, causal_check, f, global_step, linear_disentanglement_scores[-1],
                             permutation_disentanglement_scores[-1], total_loss_value, total_loss_values,
                             dep_mat, dep_loss)

            global_step += 1

        save_state_dict(args, f, "{}_f.pth".format("sup" if test else "unsup"))
        torch.cuda.empty_cache()

    report_final_disentanglement_scores(args, device, h, latent_space)


def train_step(args, data, loss, optimizer, h, test, device):
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


def train_and_log_losses(args, data, individual_losses_values, loss, optimizer, total_loss_values, h, test, device):
    if args.lr != 0:
        total_loss_value, losses_value = train_step(args, data, loss=loss, optimizer=optimizer, h=h, test=test,
                                                    device=device)
    else:
        with torch.no_grad():
            total_loss_value, losses_value = train_step(args, data, loss=loss, optimizer=optimizer, h=h, test=test,
                                                        device=device)
    total_loss_values.append(total_loss_value)
    individual_losses_values.append(losses_value)
    return total_loss_value


def log_independence_and_disentanglement(args, causal_check, global_step, h, h_ind, dep_mat, ind_test, latent_space,
                                         linear_disentanglement_scores, permutation_disentanglement_scores):
    if global_step % args.n_log_steps == 1 or global_step == args.n_steps:

        z_disentanglement = latent_space.sample_marginal(args.n_eval_samples)
        hz_disentanglement = h(z_disentanglement)

        linear_disentanglement_score, permutation_disentanglement_score = calc_disentanglement_scores(
            z_disentanglement, hz_disentanglement)
        if args.use_dep_mat:
            null_list = [False] * torch.numel(dep_mat)
            null_list[torch.argmin(dep_mat).item()] = True
            var_map = [1, 1, 2, 2]
        else:
            null_list, var_map = check_bivariate_dependence(ind_test, h_ind(z_disentanglement), hz_disentanglement)
        linear_disentanglement_scores.append(linear_disentanglement_score)
        permutation_disentanglement_scores.append(permutation_disentanglement_score)
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
        linear_disentanglement_scores.append(linear_disentanglement_scores[-1])
        permutation_disentanglement_scores.append(permutation_disentanglement_scores[-1])
        causal_check.append(causal_check[-1])
    return linear_disentanglement_scores, permutation_disentanglement_scores


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


def check_independence_z_gz(args, h_ind, ind_test, latent_space):
    z_disentanglement = latent_space.sample_marginal(args.n_eval_samples)
    linear_disentanglement_score, permutation_disentanglement_score = calc_disentanglement_scores(z_disentanglement,
                                                                                                  h_ind(
                                                                                                      z_disentanglement))
    print(f"Id. Lin. Disentanglement: {linear_disentanglement_score:.4f}")
    print(f"Id. Perm. Disentanglement: {permutation_disentanglement_score:.4f}")
    print('Run test with ground truth sources')
    if args.use_dep_mat:
        #x \times z
        dep_mat = calc_jacobian(h_ind, z_disentanglement, normalize=args.preserve_vol).abs().mean(0)
        print(dep_mat)
        null_list = [False] * torch.numel(dep_mat)
        null_list[torch.argmin(dep_mat).item()] = True
        var_map = [1, 1, 2, 2]
    else:
        null_list, var_map = check_bivariate_dependence(ind_test, h_ind(z_disentanglement), z_disentanglement)
    ######Note this is specific to a dense 2x2 triangular matrix!######
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
    print(f"{f=}")
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
    ######NOTE THAT weight_matrix_init='rvs' (used in TCL data gen in icebeem) yields linear mixing!##########
    g = invertible_network_utils.construct_invertible_mlp(
        n=args.n,
        n_layers=args.n_mixing_layer,
        act_fct=args.act_fct,
        cond_thresh_ratio=0.001,
        n_iter_cond_thresh=25000,
        lower_triangular=True,
        weight_matrix_init='rvs',
        sparsity=True,
        variant=torch.from_numpy(np.array([args.variant]))
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


def check_multivariate_dependence(ind_test: HSIC, x1:torch.Tensor, x2:torch.Tensor)->torch.Tensor:
    """
    Carries out HSIC for the multivariate case, all pairs are tested
    :param ind_test: the HSIC instance
    :param x1: tensor of the first batch of variables in the shape of (num_elem, num_dim)
    :param x2: tensor of the second batch of variables in the shape of (num_elem, num_dim)
    :return: the adjacency matrix
    """
    num_dim = x1.shape[-1]
    max_edge_num = num_dim**2
    adjacency_matrix = torch.zeros(num_dim, num_dim).bool()

    with torch.no_grad():
        for i in range(num_dim):
            for j in range(num_dim):
                adjacency_matrix[i,j] = ind_test.run_test(x1[:, i], x2[:, j], device="cpu", bonferroni=max_edge_num).item()

    return adjacency_matrix




def check_bivariate_dependence(ind_test: HSIC, x1, x2):
    decisions = []
    var_map = [1, 1, 2, 2]
    with torch.no_grad():
        decisions.append(ind_test.run_test(x1[:, 0], x2[:, 1], device="cpu", bonferroni=4).item())
        decisions.append(ind_test.run_test(x1[:, 0], x2[:, 0], device="cpu", bonferroni=4).item())
        decisions.append(ind_test.run_test(x1[:, 1], x2[:, 0], device="cpu", bonferroni=4).item())
        decisions.append(ind_test.run_test(x1[:, 1], x2[:, 1], device="cpu", bonferroni=4).item())

    return decisions, var_map


if __name__ == "__main__":

    main()
