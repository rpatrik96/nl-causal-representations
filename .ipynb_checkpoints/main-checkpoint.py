import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import os
import sys
from collections import Counter

#from data import LinearDataset
#from disentanglement import permutation_disentanglement
#from ica import ICAModel
#from train import train
from hsic import HSIC

from models.tcl.tcl_wrapper_gpu import TCL_wrapper
from data.imca import gen_TCL_data_ortho, leaky_ReLU
from metrics.mcc import mean_corr_coef
from scipy.stats import ortho_group

import argparse

def parse():
    parser = argparse.ArgumentParser(description='')
    # Monti
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--data-dim', type=int, default=2) #bivariate causal discovery
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
    parser.add_argument('--lr', type=float, default=3e-3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    os.makedirs(args.run, exist_ok=True)
    args.ckpt_dir = os.path.join(args.run, 'checkpoints')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    stepDict = {1: [int(5e3), int(5e3)], 2: [int(1e4), int(1e4)], 3: [int(1e4), int(1e4)], 4: [int(1e4), int(1e4)],
            5: [int(1e4), int(1e4)]}
    n_segments = [1,10,20,30,40,50]
    n_layers = [1,2,3,4,5]
    results = {l: {n: [] for n in n_segments} for l in n_layers}
    results_no_ica = {l: {n: [] for n in n_segments} for l in n_layers}
    results_causal = {l: {n: [] for n in n_segments} for l in n_layers}
    ind_test = HSIC(50)
    for n_layer in n_layers:
        for n_segment in n_segments:
            dat_all = gen_TCL_data_ortho(Ncomp=args.data_dim, Nsegment=n_segment, Nlayer=1, NsegmentObs=args.n_obs_per_seg, 
                                         source='Laplace', seed=args.data_seed)
            for l in range(n_layer):
                if l > 0:
                    dat_all['obs'] = leaky_ReLU(dat_all['obs'], negSlope=.2)
                A = ortho_group.rvs(args.data_dim)
                A = np.tril(A)
                dat_all['mixing'].append(A)
                dat_all['obs'] = np.dot(dat_all['obs'], A)
            x = dat_all['obs']
            y = dat_all['labels']
            s = dat_all['source']
            for seed in range(100):
                print('Running exp with L={} and n={}; seed={}'.format(n_layer, n_segment, seed))
                ckpt_folder = os.path.join(args.ckpt_dir, str(n_layer), str(n_segment), str(seed))
                res_TCL = TCL_wrapper(sensor=x.T, label=y, random_seed=seed,
                                      list_hidden_nodes=[args.data_dim * 2] * (n_layer - 1) + [args.data_dim],
                                      max_steps=stepDict[n_layer][0] * 2, max_steps_init=stepDict[n_layer][1],
                                      ckpt_dir=ckpt_folder, test=args.test)
                mcc_no_ica = mean_corr_coef(res_TCL[0].T, s ** 2)
                #Note, using fastICA, not custom ICA
                mcc_ica = mean_corr_coef(res_TCL[1].T, s ** 2)
                print('TCL mcc (no ICA): {}\t mcc: {}'.format(mcc_no_ica, mcc_ica))
                results[n_layer][n_segment].append(mcc_ica)
                results_no_ica[n_layer][n_segment].append(mcc_no_ica)
                null_1 = ind_test.run_test(x[:,0],res_TCL[1].T[:,1], device='cuda')
                null_2 = ind_test.run_test(x[:,0],res_TCL[1].T[:,0], device='cuda')
                null_3 = ind_test.run_test(x[:,1],res_TCL[1].T[:,0], device='cuda')
                null_4 = ind_test.run_test(x[:,1],res_TCL[1].T[:,1], device='cuda')
                null_list = [null_1, null_2, null_3, null_4]
                var_map = [1,1,2,2]
                if Counter([null_1, null_2, null_3, null_4]) == Counter([False, False, False, True]):
                    results_causal[n_layer][n_segment].append(True)
                    print('concluded a causal effect')
                    for i,null in enumerate(null_list):
                        if null:
                            print('cause variable is X{}'.format(str(var_map[i])))
                else:
                    results_causal[n_layer][n_segment].append(False)
            print('Proportion correct causal direction detected: {}'.format(sum(results_causal[n_layer][n_segment]) / 100))
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
