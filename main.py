import os
import sys
from collections import Counter

import numpy as np
import torch
from scipy.stats import ortho_group
from sklearn.decomposition import FastICA

from args import parse
from data.imca import gen_TCL_data_ortho, leaky_ReLU, to_one_hot
# from data import LinearDataset
# from disentanglement import permutation_disentanglement
# from ica import ICAModel
# from train import train
from hsic import HSIC
from metrics.mcc import mean_corr_coef
from models.icebeem_wrapper import ICEBEEM_wrapper
from models.ivae.ivae_wrapper import IVAE_wrapper
from models.tcl.tcl_wrapper_gpu import TCL_wrapper


def check_bivariate_dependence(x1, x2):
    decisions = []
    var_map = [1, 1, 2, 2]

    with torch.no_grad():
        decisions.append(ind_test.run_test(x1[:, 0], x2[:, 1], device=args.device, bonferroni=4))
        decisions.append(ind_test.run_test(x1[:, 0], x2[:, 0], device=args.device, bonferroni=4))
        decisions.append(ind_test.run_test(x1[:, 1], x2[:, 0], device=args.device, bonferroni=4))
        decisions.append(ind_test.run_test(x1[:, 1], x2[:, 1], device=args.device, bonferroni=4))

    return decisions, var_map


if __name__ == '__main__':

    # arguments
    args = parse()
    args.run = os.path.join(args.run, args.method)

    # file system
    os.makedirs(args.run, exist_ok=True)
    args.ckpt_dir = os.path.join(args.run, 'checkpoints')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # HSIC
    ind_test = HSIC(args.num_permutations)

    # tcl
    stepDict = {1: [int(5e3), int(5e3)],
                2: [int(1e4), int(1e4)],
                3: [int(1e4), int(1e4)],
                4: [int(1e4), int(1e4)],
                5: [int(1e4), int(1e4)]}
    n_segments = [10, 20, 30, 40, 50]
    n_layers = [1, 2, 3, 4, 5]
    results = {l: {n: [] for n in n_segments} for l in n_layers}
    results_causal = {l: {n: [] for n in n_segments} for l in n_layers}



    for n_layer in n_layers:
        for n_segment in n_segments:

            # generate data
            dat_all = gen_TCL_data_ortho(Ncomp=args.data_dim, Nsegment=n_segment, Nlayer=1,
                                         NsegmentObs=args.n_obs_per_seg,
                                         source='Laplace', seed=args.data_seed)

            # mix the sources
            for l in range(n_layer):
                if l > 0:
                    dat_all['obs'] = leaky_ReLU(dat_all['obs'], negSlope=.2)
                A = ortho_group.rvs(args.data_dim)
                A = np.tril(A)
                dat_all['mixing'].append(A)
                dat_all['obs'] = np.dot(dat_all['obs'], A)
            x = dat_all['obs']

            # prepare the labels
            if args.method == 'ivae' or args.method == 'icebeem':
                y = to_one_hot(dat_all['labels'])[0]
            else:
                y = dat_all['labels']
            s = dat_all['source']

            print('Run test with ground truth sources')

            null_list, var_map = check_bivariate_dependence(x, s)

            if Counter(null_list) == Counter([False, False, False, True]):

                print('concluded a causal effect')

                for i, null in enumerate(null_list):
                    if null:
                        print('cause variable is X{}'.format(str(var_map[i])))

            else:
                print('no causal effect...?')
                break

            for seed in range(100):

                print('Running exp with L={} and n={}; seed={}'.format(n_layer, n_segment, seed))

                if args.method == 'tcl':
                    ckpt_folder = os.path.join(args.ckpt_dir, str(n_layer), str(n_segment), str(seed))
                    res_TCL = TCL_wrapper(sensor=x.T, label=y, random_seed=seed,
                                          list_hidden_nodes=[args.data_dim * 2] * (n_layer - 1) + [args.data_dim],
                                          max_steps=stepDict[n_layer][0] * 2, max_steps_init=stepDict[n_layer][1],
                                          ckpt_dir=ckpt_folder, test=args.test)

                    mcc_no_ica = mean_corr_coef(res_TCL[0].T, s ** 2)
                    # Note, using fastICA, not custom ICA
                    mcc_ica = mean_corr_coef(res_TCL[1].T, s ** 2)
                    print('TCL mcc (no ICA): {}\t mcc: {}'.format(mcc_no_ica, mcc_ica))
                    results[n_layer][n_segment].append(max(mcc_no_ica, mcc_ica))
                    latents = [res_TCL[0].T, res_TCL[1].T][np.argmax([mcc_no_ica, mcc_ica])]

                elif args.method == 'ivae':
                    res_iVAE = IVAE_wrapper(X=x, U=y, n_layers=3, hidden_dim=args.data_dim * 2,
                                            cuda=True, max_iter=70000, lr=args.lr,
                                            ckpt_file=os.path.join(args.ckpt_dir,
                                                                   'ivae_l{}_n{}_s{}.pt'.format(n_layer, n_segment,
                                                                                                seed)), seed=seed,
                                            test=args.test)
                    elem_list = [res_iVAE[0].detach().cpu().numpy(), res_iVAE[2]['encoder'][0].detach().cpu().numpy(),
                                 FastICA().fit_transform(res_iVAE[2]['encoder'][0].detach().cpu().numpy())]
                    score_list = [mean_corr_coef(elem, s) for elem in elem_list]
                    results[n_layer][n_segment].append(max(score_list))
                    print(results[n_layer][n_segment][-1])
                    latents = elem_list[np.argmax(score_list)]

                else:
                    recov_sources = ICEBEEM_wrapper(X=x, Y=y, ebm_hidden_size=32,
                                                    n_layers_ebm=n_layer + 1, n_layers_flow=10,
                                                    lr_flow=0.00001, lr_ebm=0.0003, seed=seed,
                                                    ckpt_file=os.path.join(args.ckpt_dir,
                                                                           'icebeem_l{}_n{}_s{}.pt'.format(n_layer,
                                                                                                           n_segment,
                                                                                                           seed)),
                                                    test=args.test)
                    elem_list = [z for z in recov_sources]
                    score_list = [mean_corr_coef(elem, s) for elem in elem_list]
                    results[n_layer][n_segment].append(np.max(score_list))
                    print(results[n_layer][n_segment][-1])
                    latents = elem_list[np.argmax(score_list)]



                null_list, var_map = check_bivariate_dependence(x, latents)

                if Counter(null_list) == Counter([False, False, False, True]):
                    results_causal[n_layer][n_segment].append(True)
                    print('concluded a causal effect')

                    for i, null in enumerate(null_list):
                        if null:
                            print('cause variable is X{}'.format(str(var_map[i])))
                else:
                    results_causal[n_layer][n_segment].append(False)

            print('Proportion correct causal direction detected: {}'.format(
                sum(results_causal[n_layer][n_segment]) / 100))

    fname = os.path.join(args.run, 'fig2_top.p')
    pickle.dump(r, open(fname, "wb"))
    sys.exit()
    #######################
    """CONSTANTS"""
    # ICA
    DIM = args.dim
    SIGNAL_MODEL = torch.distributions.Laplace(args.loc, args.scale)

    # Dataset
    NUM_SAMPLES = args.num_samples
    # todo: test with 0 values
    A = args.a_var
    B = args.b_var
    C = args.c_var

    # Training
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LR = args.lr

    ds = LinearDataset(A, B, C, NUM_SAMPLES)
    ica = ICAModel(DIM, SIGNAL_MODEL)

    dl = DataLoader(ds, BATCH_SIZE, True)

    optim = torch.optim.SGD(ica.parameters(), LR)

    losses, neg_entropies, dets = train(ica, dl, optim, NUM_EPOCHS)

    # ML formulation losses
    plt.plot(losses, label="Loss")
    plt.plot(neg_entropies, label="Entropy")
    plt.plot(dets, label="Det loss")
    plt.legend()

    # mcc
    (mcc, mat), data = permutation_disentanglement(ds.noise.T, ds.data @ ica.W.data,
                                                   mode="spearman", solver="munkres")
    print(f"MCC={mcc:.4f}")
    data = torch.tensor(data)

    hsic = HSIC(50)
    hsic.run_test(data, ds.noise.T)
