import os
import random

import numpy as np
import torch


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


def setup_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)


def save_state_dict(args, model, pth="g.pth"):
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model.state_dict(), os.path.join(args.save_dir, pth))


def print_statistics(args, causal_check, f, global_step, linear_disentanglement_score,
                     permutation_disentanglement_score, total_loss_value, total_loss_values,
                     dep_mat, dep_loss):
    if global_step % args.n_log_steps == 1 or global_step == args.n_steps:
        print(
            f"Step: {global_step} \t",
            f"Loss: {total_loss_value:.4f} \t",
            f"<Loss>: {np.mean(np.array(total_loss_values[-args.n_log_steps:])):.4f} \t",
            f"Lin. Disentanglement: {linear_disentanglement_score:.4f} \t",
            f"Perm. Disentanglement: {permutation_disentanglement_score:.4f}",
            f"Causal. Check: {causal_check[-1]:.4f}",
        )
        print(dep_mat.detach())
        print("dep Loss: {}".format(dep_loss))
        if args.normalization == "learnable_sphere":
            print(f"r: {f[-1].r}")


def set_learning_mode(args):
    if args.mode == 'unsupervised':
        learning_modes = [False]
    elif args.mode == 'supervised':
        learning_modes = [True]
    else:
        learning_modes = [True, False]

    args.learning_modes = learning_modes


def set_device(args) -> None:
    device = "cuda"
    if not torch.cuda.is_available() or args.no_cuda is True:
        device = "cpu"

    print(f"{device=}")

    args.device = device
