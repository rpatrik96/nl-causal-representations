import argparse


def parse():

    parser = argparse.ArgumentParser(description='')

    # general
    parser.add_argument('--device', type=str, default='cuda', help='Device type.')

    # Monti
    parser.add_argument('--num-permutations', type=int, default=50)
    parser.add_argument('--method', type=str, default='tcl',
                        help='Method to employ. Should be TCL, iVAE or ICE-BeeM')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--data-dim', type=int, default=2)  # bivariate causal discovery
    parser.add_argument('--n-segments', type=int, default=1)
    parser.add_argument('--n-obs-per-seg', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=1)
    parser.add_argument('--data-seed', type=int, default=1)
    parser.add_argument('--run', type=str, default='run/', help='Path for saving running related data.')

    # ICA
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--loc', type=float, default=0.)
    parser.add_argument('--scale', type=float, default=1.)

    # Dataset
    parser.add_argument('--num-samples', type=int, default=2048)
    parser.add_argument('--a-var', type=float, default=0.65)
    parser.add_argument('--b-var', type=float, default=-1.15)
    parser.add_argument('--c-var', type=float, default=0.5)

    # Training
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()

    return args