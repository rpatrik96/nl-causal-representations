from args import parse_args
from cl_ica import latent_spaces
from indep_check import IndependenceChecker
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
from prob_utils import setup_marginal, setup_conditional, check_independence_z_gz
from runner import Runner
from utils import setup_seed, save_state_dict, set_learning_mode, set_device


def main():
    # setup
    args = parse_args()

    set_device(args)
    setup_seed(args.seed)
    set_learning_mode(args)

    runner = Runner(args)


    indep_checker = IndependenceChecker(args)

    # distributions
    latent_space = latent_spaces.LatentSpace(space=(runner.model.space), sample_marginal=(setup_marginal(args)),
                                             sample_conditional=(setup_conditional(args)), )

    check_independence_z_gz(indep_checker, runner.model.h_ind, latent_space)

    save_state_dict(args, runner.model.decoder)

    runner.training_loop(indep_checker, latent_space)


if __name__ == "__main__":
    main()
