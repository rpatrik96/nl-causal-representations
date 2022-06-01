from os.path import isfile

import numpy as np
import pandas as pd

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

                true_unmixing_jacobians.append(
                    np.array(summary["Mixing/unmixing_jacobian"])
                )
                est_unmixing_jacobians.append(
                    np.array(summary["Unmixing/unmixing_jacobian"])
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
