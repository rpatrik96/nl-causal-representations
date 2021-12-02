import os

import pip


def install_package():
    """
    Install the current package to ensure that imports work.
    """
    try:
        import care_nl_ica
    except:
        print("Package not installed, installing...")
        pip.main(["install", f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}", "--upgrade"])


def main():
    # install the package
    install_package()

    # setup
    from args import parse_args
    args = parse_args()

    from runner import Runner
    from utils import setup_seed, save_state_dict, set_learning_mode, set_device
    set_device(args)
    setup_seed(args.seed)
    set_learning_mode(args)

    runner = Runner(args)

    save_state_dict(args, runner.model.decoder)

    runner.training_loop()


if __name__ == "__main__":
    main()
