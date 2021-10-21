import os
import pip


def install_package():
    """
    Install the current package to ensure that imports work.
    """

    pip.main(["install", f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}", "--upgrade"])



def main():

    # install the package
    install_package()

    # setup
    from args import parse_args
    args = parse_args()

    from prob_utils import setup_marginal, setup_conditional, check_independence_z_gz
    from runner import Runner
    from utils import setup_seed, save_state_dict, set_learning_mode, set_device
    set_device(args)
    setup_seed(args.seed)
    set_learning_mode(args)

    runner = Runner(args)

    from indep_check import IndependenceChecker
    indep_checker = IndependenceChecker(args)

    # distributions
    from cl_ica import latent_spaces
    latent_space = latent_spaces.LatentSpace(space=(runner.model.space), sample_marginal=(setup_marginal(args)),
                                             sample_conditional=(setup_conditional(args)), )

    dep_mat = check_independence_z_gz(indep_checker, runner.model.decoder, latent_space)

    if args.use_flows:
        runner.model.encoder.confidence.inject_structure(dep_mat, args.inject_structure)

    save_state_dict(args, runner.model.decoder)

    runner.training_loop(indep_checker, latent_space)


if __name__ == "__main__":
    main()
