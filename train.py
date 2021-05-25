import torch
from torch.utils.data import DataLoader

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
        for batch in dataloader:

            neg_entropy, det = model.loss(batch)

            neg_entropies.append(neg_entropy.item())
            dets.append(det.item())

            loss = neg_entropy + det

            losses.append(loss.item())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return losses, neg_entropies, dets
