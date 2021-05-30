import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data import LinearDataset
from disentanglement import permutation_disentanglement
from ica import ICAModel
from train import train
from hsic import HSIC

if __name__ == '__main__':
    """CONSTANTS"""
    # ICA
    DIM = 3
    SIGNAL_MODEL = torch.distributions.Laplace(0, 1)

    # Dataset
    NUM_SAMPLES = 2048
    # todo: test with 0 values
    A = .65
    B = -1.15
    C = 0.5

    # Training
    BATCH_SIZE = 128
    NUM_EPOCHS = 4000
    LR = 3e-3

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
