from os.path import isfile

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import hamming

import wandb
from care_nl_ica.metrics.dep_mat import (
    correct_jacobian_permutations,
    jacobian_edge_accuracy,
    JacobianBinnedPrecisionRecall,
)
from care_nl_ica.models.sinkhorn import learn_permutation

BLUE = "#1A85FF"
RED = "#D0021B"
METRIC_EPS = 1e-6

from matplotlib import pyplot as plt
from matplotlib import rc


def plot_typography(
    usetex: bool = False, small: int = 16, medium: int = 20, big: int = 22
):
    """
    Initializes font settings and visualization backend (LaTeX or standard matplotlib).
    :param usetex: flag to indicate the usage of LaTeX (needs LaTeX indstalled)
    :param small: small font size in pt (for legends and axes' ticks)
    :param medium: medium font size in pt (for axes' labels)
    :param big: big font size in pt (for titles)
    :return:
    """

    # font family
    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})

    # backend
    rc("text", usetex=usetex)
    rc("font", family="serif")

    # font sizes
    rc("font", size=small)  # controls default text sizes
    rc("axes", titlesize=big)  # fontsize of the axes title
    rc("axes", labelsize=medium)  # fontsize of the x and y labels
    rc("xtick", labelsize=small)  # fontsize of the tick labels
    rc("ytick", labelsize=small)  # fontsize of the tick labels
    rc("legend", fontsize=small)  # legend fontsize
    rc("figure", titlesize=big)  # fontsize of the figure title


def sweep2df(
    sweep_runs,
    filename,
    save=False,
    load=False,
    entity="causal-representation-learning",
    project="nl-causal-representations",
):
    csv_name = f"{filename}.csv"
    npy_name = f"{filename}.npz"

    if load is True and isfile(csv_name) is True and isfile(npy_name) is True:
        print(f"\t Loading {filename}...")
        npy_data = np.load(npy_name)
        true_unmixing_jacobians = npy_data["true_unmixing_jacobians"]
        est_unmixing_jacobians = npy_data["est_unmixing_jacobians"]
        permute_indices = npy_data["permute_indices"]
        try:
            hsic_adj = npy_data["hsic_adj"]
        except:
            hsic_adj = None
        try:
            munkres_permutation_indices = npy_data["munkres_permutation_indices"]
        except:
            munkres_permutation_indices = None

        return pd.read_csv(csv_name), (
            true_unmixing_jacobians,
            est_unmixing_jacobians,
            permute_indices,
            hsic_adj,
            munkres_permutation_indices,
        )
    data = []
    true_unmixing_jacobians = []
    est_unmixing_jacobians = []
    permute_indices = []
    munkres_permutation_indices = []
    hsic_adj = []
    max_dim = -1
    for run in sweep_runs:

        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary = run.summary._json_dict

        if run.state == "finished":
            try:
                # if True:
                # .config contains the hyperparameters.
                #  We remove special values that start with _.
                config = {k: v for k, v in run.config.items() if not k.startswith("_")}

                dim = config["latent_dim"]
                permute = config["permute"]
                variant = config["variant"]
                n_mixing_layer = config["n_mixing_layer"]
                use_sem = config["use_sem"]
                nonlin_sem = config["nonlin_sem"]
                force_chain = config["force_chain"]
                force_uniform = config["force_uniform"]

                mcc = summary["val_mcc"]
                val_loss = summary["val_loss"]

                est_unmixing_jacobians.append(
                    np.array(summary["Unmixing/unmixing_jacobian"])
                    if dim <= 5
                    else run.logged_artifacts()[0]
                    .get("dep_mat_table")
                    .get_column("dep_mat", "numpy")
                    .reshape(dim, dim)
                )
                true_unmixing_jacobians.append(
                    np.array(summary["Mixing/unmixing_jacobian"])
                    if dim <= 5
                    else run.logged_artifacts()[1]
                    .get("gt_unmixing_jacobian_table")
                    .get_column("gt_unmixing_jacobian", "numpy")
                    .reshape(dim, dim)
                )

                if "Val/hsic_adj" in summary.keys():
                    if dim <= 5:
                        hsic_adj.append(np.array(summary["Val/hsic_adj"]))
                    else:
                        s = f"{entity}/{project}/run-{run.id}-hsic_adj_table:latest"

                        run = wandb.init()
                        hsic_adj.append(
                            run.use_artifact(s, type="run_table")
                            .get("hsic_adj_table")
                            .get_column("hsic_adj", "numpy")
                            .reshape(dim, dim)
                        )

                if "munkres_permutation_idx" in summary.keys():
                    munkres_permutation_indices.append(
                        summary["munkres_permutation_idx"]
                    )

                permute_indices.append(
                    summary["Mixing/permute_indices"]
                    if permute is True
                    else np.arange(0, dim)
                )

                if dim > max_dim:
                    max_dim = dim

                data.append(
                    [
                        run.name,
                        dim,
                        permute,
                        variant,
                        n_mixing_layer,
                        use_sem,
                        nonlin_sem,
                        force_chain,
                        force_uniform,
                        mcc,
                        val_loss,
                    ]
                )
            except:
                print(f"Encountered a faulty run with ID {run.name}")

    runs_df = pd.DataFrame(
        data,
        columns=[
            "name",
            "dim",
            "permute",
            "variant",
            "n_mixing_layer",
            "use_sem",
            "nonlin_sem",
            "force_chain",
            "force_uniform",
            "mcc",
            "val_loss",
        ],
    ).fillna(0)

    if save is True:
        runs_df.to_csv(csv_name)
        np.savez_compressed(
            npy_name,
            true_unmixing_jacobians=true_unmixing_jacobians,
            est_unmixing_jacobians=est_unmixing_jacobians,
            permute_indices=permute_indices,
            hsic_adj=hsic_adj,
            munkres_permutation_indices=munkres_permutation_indices,
        )

    return runs_df, (
        true_unmixing_jacobians,
        est_unmixing_jacobians,
        permute_indices,
        hsic_adj,
        munkres_permutation_indices,
    )


def format_violin(vp, facecolor=BLUE):
    for el in vp["bodies"]:
        el.set_facecolor(facecolor)
        el.set_edgecolor("black")
        el.set_linewidth(0.75)
        el.set_alpha(0.9)
    for pn in ["cbars", "cmins", "cmaxes", "cmedians"]:
        vp_ = vp[pn]
        vp_.set_edgecolor("black")
        vp_.set_linewidth(0.5)


import matplotlib.pyplot as plt


def create_violinplot(groups, xlabel, ylabel, xticklabels, filename=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax = ax.twinx()

    vp = ax.violinplot(groups, showmedians=True)
    format_violin(vp, BLUE)

    ax.set_xticklabels(xticklabels)
    # ax.set_xticks(xticks)
    # plt.locator_params(axis='y', nbins=5)
    # plt.yticks(fontsize=24)
    # plt.ylim([0, 0.5])
    ax.set_ylabel(ylabel)
    # ax.set_xlabel(xlabel)
    if filename is not None:
        plt.savefig(f"{filename}.svg")
    return ax


def violin_by_prior(
    gauss_data,
    laplace_data,
    uniform_data,
    xticks,
    xlabel,
    ylabel,
    offset,
    filename,
    figsize=(8, 6),
    log=False,
):
    plt.figure(figsize=figsize)
    vp_gauss = plt.violinplot(
        [np.log10(i) if log is True else i for i in gauss_data], positions=xticks
    )
    vp_laplace = plt.violinplot(
        [np.log10(i) if log is True else i for i in laplace_data],
        positions=-offset + xticks,
    )
    vp_uniform = plt.violinplot(
        [np.log10(i) if log is True else i for i in uniform_data],
        positions=offset + xticks,
    )
    plt.legend(
        [vp_gauss["bodies"][0], vp_laplace["bodies"][0], vp_uniform["bodies"][0]],
        ["gaussian", "laplace", "uniform"],
        loc="upper right",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xticks)
    # plt.tight_layout()
    plt.savefig(filename)


def learning_stats(
    df,
    true_unmix_jacobians,
    est_unmix_jacobians,
    permute_indices,
    hamming_threshold=1e-2,
    selector_col="nonlin_sem",
    weight_threshold=None,
    dag_permute=True,
    num_steps=5000,
    rank_acc=False,
    drop_smallest=True,
    binary=True,
):
    for dim in df.dim.unique():
        for selector in df[selector_col].unique():
            success = []
            hamming = []
            accuracy = []
            permutation_losses_dag = []
            permutation_losses_ica = []
            for (selector_item, j_gt, j_est, permute) in zip(
                df[selector_col],
                true_unmix_jacobians,
                est_unmix_jacobians,
                permute_indices,
            ):
                if j_gt.shape[0] == dim and selector_item == selector:
                    s, h, a, p_dag, p_ica = learn_permutation(
                        j_gt,
                        j_est,
                        permute,
                        num_steps=num_steps,
                        tril_weight=10.0,
                        triu_weigth=20.0,
                        diag_weight=6.0,
                        lr=1e-4,
                        verbose=False,
                        drop_smallest=drop_smallest,
                        threshold=weight_threshold,
                        binary=binary,
                        hamming_threshold=hamming_threshold,
                        dag_permute=dag_permute,
                        rank_acc=rank_acc,
                    )

                    print(s, p_dag, p_ica)
                    if s >= 0:
                        success.append(s)
                        hamming.append(h)
                        accuracy.append(a)
                        permutation_losses_dag.append(p_dag)
                        permutation_losses_ica.append(p_ica)

            mcc = df.mcc[(df.dim == dim) & (df[selector_col] == selector)]
            print("----------------------------------")
            if len(success) > 0:
                print(
                    f"{dim=} ({selector_col}={selector})\tMCC={mcc.mean():.3f}+{mcc.std():.3f}\tAcc(order):{np.array(success).mean():.3f}\t  Acc:{np.array(accuracy).mean():.3f}\tSHD:{np.array(hamming).mean():.6f}\t[{len(success)} items]"
                )
            else:
                print(f"No experiments for {dim=} ({selector_col}={selector})")


def perm2matrix(permute_indices):
    m = torch.zeros(l := len(permute_indices), l)
    m[list(range(l)), permute_indices] = 1.0

    return m


def corrected_jacobian_stats(
    df,
    true_unmix_jacobians,
    est_unmix_jacobians,
    permute_indices,
    hsic_adj,
    munkres_indices,
    hamming_threshold=1e-3,
    selector_col="nonlin_sem",
) -> dict:
    j_hamming = lambda gt, est: hamming(
        gt.abs().reshape(
            -1,
        )
        > hamming_threshold,
        est.detach()
        .abs()
        .reshape(
            -1,
        )
        > hamming_threshold,
    )
    stats: dict = dict()
    for dim in df.dim.unique():
        stats[dim] = dict()

        for selector in df[selector_col].unique():
            if hsic_adj is None:
                hsic_adj = [None] * len(true_unmix_jacobians)

            jac_prec_recall = JacobianBinnedPrecisionRecall(50, log_base=10, start=-7)
            # success = []
            hamming_dist = []
            accuracy = []
            accuracy_hsic = [] if hsic_adj[0] is not None else -1.0
            precision_hsic = [] if hsic_adj[0] is not None else -1.0
            recall_hsic = [] if hsic_adj[0] is not None else -1.0
            for (selector_item, j_gt, j_est, p, hsic, munkres) in zip(
                df[selector_col],
                true_unmix_jacobians,
                est_unmix_jacobians,
                permute_indices,
                hsic_adj,
                munkres_indices,
            ):
                if j_gt.shape[0] == dim and selector_item == selector:
                    j_est = torch.from_numpy(j_est.astype(np.float32))
                    j_gt = torch.from_numpy(j_gt.astype(np.float32))
                    p = perm2matrix(p)
                    munkres = perm2matrix(munkres)

                    j_est_corr = correct_jacobian_permutations(j_est, munkres, p)
                    jac_prec_recall.update(j_est_corr, j_gt)

                    accuracy.append(jacobian_edge_accuracy(j_est_corr, j_gt))

                    if hsic is not None:
                        hsic = torch.from_numpy(hsic.astype(np.float32))
                        hsic_corr = correct_jacobian_permutations(hsic, munkres, p)
                        hsic_edges = hsic2edge(hsic_corr)

                        numel = dim**2
                        hsic_tp = (hsic_edges == j_gt.bool()).float().sum()
                        hsic_fp = (
                            (hsic_edges == j_gt.bool().logical_not()).float().sum()
                        )
                        hsic_fn = (
                            (hsic_edges.logical_not() == j_gt.bool()).float().sum()
                        )

                        precision_hsic.append(
                            (hsic_tp + METRIC_EPS) / (hsic_tp + hsic_fp + METRIC_EPS)
                        )
                        recall_hsic.append(
                            (hsic_tp + METRIC_EPS) / (hsic_tp + hsic_fn + METRIC_EPS)
                        )

                        accuracy_hsic.append(hsic_tp / numel)
                    hamming_dist.append(j_hamming(j_gt, j_est_corr))

            precisions, recalls, thresholds = jac_prec_recall.compute()
            mcc = df.mcc[(df.dim == dim) & (df[selector_col] == selector)]
            accuracy = np.array(accuracy)
            accuracy_hsic = np.array(accuracy_hsic)
            precision_hsic = np.array(precision_hsic)
            recall_hsic = np.array(recall_hsic)
            hamming_dist = np.array(hamming_dist)
            print("----------------------------------")
            if len(accuracy) > 0:
                print(
                    f"{dim=} ({selector_col}={selector})\tMCC={mcc.mean():.3f}+{mcc.std():.3f}\t  Acc:{accuracy.mean():.3f}+{accuracy.std():.3f}\tAcc (HSIC):{accuracy_hsic.mean():.3f}+{accuracy_hsic.std():.3f}\tPrec (HSIC):{precision_hsic.mean():.3f}+{precision_hsic.std():.3f}\tRec (HSIC):{recall_hsic.mean():.3f}+{recall_hsic.std():.3f}\tSHD:{hamming_dist.mean():.6f}+{hamming_dist.std():.6f}\t[{len(accuracy)} items]"
                )
            else:
                print(f"No experiments for {dim=} ({selector_col}={selector})")

            stats[dim][selector] = {
                "precisions": precisions,
                "recalls": recalls,
                "thresholds": thresholds,
                "accuracy": accuracy,
                "accuracy_hsic": accuracy_hsic,
            }
    return stats


def hsic2edge(hsic):
    dim = hsic.shape[0]
    hsic_edges = torch.eye(dim)

    for i in range(dim):
        for j in range(i + 1, dim):
            if hsic[i, j] + hsic[i, i] + hsic[j, i] + hsic[j, j] == 1:
                # as the test is ran for s_i, x_j, the (j,i) edge should be set
                if hsic[j, i] == 1:
                    hsic_edges[i, j] = 1
                elif hsic[i, j] == 1:
                    hsic_edges[j, i] = 1

    # due to iteration pattern, we need to transpose hsic_edges
    return hsic_edges.T


# Source: https://matplotlib.org/3.1.1/gallery/specialty_plots/hinton_demo.html


def hinton(matrix, max_weight=None, ax=None, filename=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor("gray")
    ax.set_aspect("equal", "box")
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = BLUE if w > 0 else RED
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle(
            [x - size / 2, y - size / 2], size, size, facecolor=color, edgecolor=color
        )
        ax.add_patch(rect)

    ax.autoscale_view()
    if filename is not None:
        plt.savefig(f"{filename}.svg")
