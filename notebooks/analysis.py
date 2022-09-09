from os.path import isfile

import numpy as np
import pandas as pd
import torch

from care_nl_ica.models.sinkhorn import learn_permutation
from care_nl_ica.metrics.dep_mat import (
    correct_ica_scale_permutation,
    jacobian_edge_accuracy,
    JacobianBinnedPrecisionRecall,
)

BLUE = "#1A85FF"
RED = "#D0021B"


def sweep2df(sweep_runs, filename, save=False, load=False):
    csv_name = f"{filename}.csv"
    npy_name = f"{filename}"
    if load is True and isfile(csv_name) is True and isfile(npy_name) is True:
        print(f"\t Loading {filename}...")
        npy_data = np.load(npy_name)
        true_unmixing_jacobians = npy_data["true_unmixing_jacobians"]
        est_unmixing_jacobians = npy_data["est_unmixing_jacobians"]
        permute_indices = npy_data["permute_indices"]
        return pd.read_csv(filename), (
            true_unmixing_jacobians,
            est_unmixing_jacobians,
            permute_indices,
        )
    data = []
    true_unmixing_jacobians = []
    est_unmixing_jacobians = []
    permute_indices = []
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
                permute_indices.append(summary["Mixing/permute_indices"])

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
        )

    return runs_df, (true_unmixing_jacobians, est_unmixing_jacobians, permute_indices)


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
):

    for dim in df.dim.unique():
        for selector in df[selector_col].unique():
            success = []
            hamming = []
            accuracy = []
            for (selector_item, j_gt, j_est, permute) in zip(
                df[selector_col],
                true_unmix_jacobians,
                est_unmix_jacobians,
                permute_indices,
            ):
                if j_gt.shape[0] == dim and selector_item == selector:
                    s, h, a = learn_permutation(
                        j_gt,
                        j_est,
                        permute,
                        triu_weigth=20.0,
                        tril_weight=10.0,
                        diag_weight=6.0,
                        num_steps=num_steps,
                        lr=1e-4,
                        verbose=True,
                        drop_smallest=True,
                        threshold=weight_threshold,
                        binary=True,
                        hamming_threshold=hamming_threshold,
                        dag_permute=dag_permute,
                    )

                    success.append(s)
                    hamming.append(h)
                    accuracy.append(a)

            mcc = df.mcc[(df.dim == dim) & (df[selector_col] == selector)]
            print("----------------------------------")
            print("----------------------------------")
            if len(success) > 0:
                print(
                    f"{dim=} ({selector_col}={selector})\tMCC={mcc.mean():.3f}+{mcc.std():.3f}\tAcc(order):{np.array(success).mean():.3f}\t  Acc:{np.array(accuracy).mean():.3f}\tSHD:{np.array(hamming).mean():.6f}\t[{len(success)} items]"
                )
            else:
                print(f"No experiments for {dim=} ({selector_col}={selector})")
            print("----------------------------------")
            print("----------------------------------")


def corrected_jacobian_stats(
    df,
    true_unmix_jacobians,
    est_unmix_jacobians,
    hamming_threshold=1e-2,
    selector_col="nonlin_sem",
    weight_threshold=None,
    dag_permute=True,
):

    for dim in df.dim.unique():
        for selector in df[selector_col].unique():
            jac_prec_recall = JacobianBinnedPrecisionRecall(15)
            # success = []
            # hamming = []
            accuracy = []
            for (selector_item, j_gt, j_est) in zip(
                df[selector_col],
                true_unmix_jacobians,
                est_unmix_jacobians,
            ):
                if j_gt.shape[0] == dim and selector_item == selector:
                    j_est = torch.from_numpy(j_est)
                    j_gt = torch.from_numpy(j_gt)
                    j_est_corr = correct_ica_scale_permutation(j_est, j_gt)
                    acc = jacobian_edge_accuracy(j_est_corr, j_gt)
                    jac_prec_recall.update(j_est_corr, j_gt)

                    # success.append(s)
                    # hamming.append(h)
                    accuracy.append(acc)
            precisions, recalls, thresholds = jac_prec_recall.compute()
            print(precisions)
            print(recalls)
            print(thresholds)
            mcc = df.mcc[(df.dim == dim) & (df[selector_col] == selector)]
            print("----------------------------------")
            print("----------------------------------")
            if len(accuracy) > 0:
                print(
                    f"{dim=} ({selector_col}={selector})\tMCC={mcc.mean():.3f}+{mcc.std():.3f}\t  Acc:{np.array(accuracy).mean():.3f}"  # \tSHD:{np.array(hamming).mean():.6f}\t[{len(success)} items]"
                )
            else:
                print(f"No experiments for {dim=} ({selector_col}={selector})")
            print("----------------------------------")
            print("----------------------------------")
