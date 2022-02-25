import argparse

import dep_mat


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Disentanglement with InfoNCE/Contrastive Learning - MLP Mixing"
    )
    parser.add_argument("--verbose", action="store_true", help="Print out details")
    parser.add_argument("--use-flows", action="store_true", help="Use a Flow encoder")
    parser.add_argument(
        "--use-ar-mlp", action="store_true", help="Use the AR MLP encoder"
    )
    parser.add_argument(
        "--use-reverse",
        action="store_true",
        help="Use reverse layers in the Flow encoder",
    )
    parser.add_argument(
        "--use-batch-norm",
        action="store_true",
        help="Use batchnorm layers in the Flow encoder",
    )
    parser.add_argument(
        "--log-latent-rec",
        action="store_true",
        help="Log the latents and their reconstructions",
    )
    parser.add_argument(
        "--triangular",
        action="store_true",
        help="Force the AR MLP bottleneck to be triangular",
    )
    parser.add_argument(
        "--triangularity-loss",
        type=float,
        default=0.0,
        help="triangularity loss on the correlation matrix",
    )
    parser.add_argument(
        "--qr-loss", type=float, default=0.0, help="QR loss on the bottleneck matrix"
    )
    parser.add_argument(
        "--cholesky-permutation",
        action="store_true",
        help="Estimate the permutation matrix from the Cholesky decomposition of the Jacobian",
    )
    parser.add_argument("--use-sem", action="store_true", help="Use SEM as decoder")
    parser.add_argument(
        "--sinkhorn", action="store_true", help="Use the Sinkhorn network"
    )
    parser.add_argument("--permute", action="store_true", help="Learn the permutation")
    parser.add_argument(
        "--budget",
        type=float,
        default=0.0,
        help="Constrain the non-zero elements on the bottleneck",
    )
    parser.add_argument(
        "--normalize-latents",
        action="store_true",
        help="Normalizes the latent (marginal) distribution",
    )
    parser.add_argument(
        "--nonlin-sem", action="store_true", help="Use nonlinear SEM as decoder"
    )
    parser.add_argument(
        "--use-bias", action="store_true", help="Use bias in the network"
    )
    parser.add_argument("--l1", type=float, default=0.0, help="L1 regularization")
    parser.add_argument("--l2", type=float, default=0.0, help="L2 regularization")
    parser.add_argument(
        "--entropy-coeff",
        default=0.0,
        type=float,
        help="Entropy coefficient on the Sinkhorn weights",
    )
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument(
        "--start-step",
        type=int,
        default=None,
        help="Starting step index to activate functions",
    )
    parser.add_argument(
        "--use-dep-mat", action="store_true", help="Use the dependency matrix"
    )
    parser.add_argument(
        "--inject-structure",
        action="store_true",
        help="Injects a fixed structure into the flow to see the effect when the GT cannot be recovered",
    )
    parser.add_argument(
        "--preserve-vol",
        action="store_true",
        help="Normalize the dependency matrix to have determinant=1",
    )
    parser.add_argument(
        "--learnable-mask",
        action="store_true",
        help="Makes the masks in the flow learnable",
    )
    parser.add_argument("--num-permutations", type=int, default=50)
    parser.add_argument("--n-eval-samples", type=int, default=512)
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
    parser.add_argument(
        "--alpha",
        default=0.5,
        type=float,
        help="Weight factor between the two loss components",
    )
    parser.add_argument(
        "--normalization",
        choices=("", "fixed_box", "learnable_box", "fixed_sphere", "learnable_sphere"),
        help="Output normalization to use. If empty, do not normalize at all.",
        default="",
    )
    parser.add_argument("--mode", type=str, default="unsupervised")
    parser.add_argument(
        "--data-gen-mode", type=str, default="rvs", choices=["rvs", "pcl"]
    )
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
    parser.add_argument("--n-steps", type=int, default=1001)
    parser.add_argument("--resume-training", action="store_true")

    # W and B
    parser.add_argument(
        "--use-wandb", action="store_true", help="Log with Weights&Biases"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="experiment",
        help="This is the name of the experiment on Weights and Biases",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Notes for the run on Weights and Biases",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",  # 0 or more values expected => creates a list
        default=None,
        help="Tags for the run on Weights and Biases",
    )

    args = parser.parse_args(args)

    add_tags(args)

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    return args


def add_tags(args):
    if args.tags is None:
        args.tags = []

    if args.use_sem is True:
        args.tags.append("sem")

    if args.nonlin_sem is True:
        args.tags.append("nonlinear")
    else:
        args.tags.append("linear")

    if args.sinkhorn is True:
        args.tags.append("sinkhorn")

    if args.permute is True:
        args.tags.append("permute")

    if args.use_ar_mlp is False:
        args.tags.append("mlp")
    else:
        args.tags.append("bottleneck")
        if args.triangular is True:
            args.tags.append("triangular")

    if args.use_flows is True:
        args.tags.append("flows")

    if args.normalize_latents is True:
        args.tags.append("normalization")

    if args.l1 != 0.0:
        args.tags.append(f"L1")

    if args.l2 != 0.0:
        args.tags.append(f"L2")

    if args.triangularity_loss != 0.0:
        args.tags.append(f"triangularity")

    if args.entropy_coeff != 0.0:
        args.tags.append(f"entropy")

    if args.qr_loss != 0.0:
        args.tags.append(f"QR")

    args.tags = list(set(args.tags))
