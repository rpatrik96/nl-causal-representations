import os
import random
from typing import Dict, Literal

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


def set_learning_mode(args):
    if args.mode == "unsupervised":
        learning_modes = [False]
    elif args.mode == "supervised":
        learning_modes = [True]
    else:
        learning_modes = [True, False]

    args.learning_modes = learning_modes


def set_device(args) -> None:
    device = "cuda"
    if not torch.cuda.is_available() or args.no_cuda is True:
        device = "cpu"

    if args.verbose is True:
        print(f"{device=}")

    args.device = device


def matrix_to_dict(matrix, name, panel_name=None, triangular=False) -> Dict[str, float]:
    if matrix is not None:
        if triangular is False:
            labels = [
                f"{name}_{i}{j}"
                if panel_name is None
                else f"{panel_name}/{name}_{i}{j}"
                for i in range(matrix.shape[0])
                for j in range(matrix.shape[1])
            ]
        else:
            labels = [
                f"{name}_{i}{j}"
                if panel_name is None
                else f"{panel_name}/{name}_{i}{j}"
                for i in range(matrix.shape[0])
                for j in range(i + 1)
            ]
        data = (
            matrix.detach()
            .cpu()
            .reshape(
                -1,
            )
            .tolist()
        )

    return {key: val for key, val in zip(labels, data)}


OutputNormalizationType = Literal[
    "", "fixed_box", "learnable_box", "fixed_sphere", "learnable_sphere"
]
SpaceType = Literal["box", "sphere", "unbounded"]
DataGenType = Literal["rvs", "pcl"]


def add_tags(args):
    try:
        args.tags
    except:
        args.tags = []

    if args.tags is None:
        args.tags = []

    if args.data.use_sem is True:
        args.tags.append("sem")

    if args.data.nonlin_sem is True:
        args.tags.append("nonlinear")
    else:
        args.tags.append("linear")

    if args.model.sinkhorn is True:
        args.tags.append("sinkhorn")

    if args.data.permute is True:
        args.tags.append("permute")

    if args.model.use_ar_mlp is False:
        args.tags.append("mlp")
    else:
        args.tags.append("bottleneck")
        if args.model.triangular is True:
            args.tags.append("triangular")

    if args.model.use_flows is True:
        args.tags.append("flows")

    if args.model.normalize_latents is True:
        args.tags.append("normalization")

    if args.model.l1 != 0.0:
        args.tags.append(f"L1")

    if args.model.entropy != 0.0:
        args.tags.append(f"entropy")

    if args.model.qr != 0.0:
        args.tags.append(f"QR")

    return list(set(args.tags))


def get_cuda_stats(where):
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(0))
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    print(where)

    print(f"total    : {info.total//1024**2}")
    print(f"free     : {info.free//1024**2}")
    print(f"used     : {info.used//1024**2}")
