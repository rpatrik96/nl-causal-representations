import torch
from torch.utils.data import DataLoader
import numpy as np

from ica import ICAModel


def train(model: ICAModel, dataloader: DataLoader, optimizer: torch.optim.Optimizer, num_epochs:int):
    """
    Training loop

    :param model: the ICA model
    :param dataloader: DataLoader
    :param optimizer: optimizer
    """

    losses = []
    neg_entropies = []
    dets = []

    for i in range(num_epochs):
        print('Epoch: {}'.format(i))
        ep_losses = []
        for batch in dataloader:

            neg_entropy, det = model.loss(batch)

            neg_entropies.append(neg_entropy.item())
            dets.append(det.item())

            loss = neg_entropy + det

            losses.append(loss.item())
            ep_losses.append(loss.item())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Loss: {}'.format(np.mean(ep_losses)))

    return losses, neg_entropies, dets
