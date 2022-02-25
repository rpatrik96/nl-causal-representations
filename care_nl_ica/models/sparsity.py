import torch
import torch.nn as nn


class SparseBudgetNet(nn.Module):
    def __init__(
        self,
        num_dim: int,
    ):
        super(SparseBudgetNet, self).__init__()
        self.num_dim = num_dim
        self.budget: int = self.num_dim * (self.num_dim + 1) // 2
        self.weight = nn.Parameter(
            nn.Linear(self.num_dim, self.num_dim).weight, requires_grad=True
        )

    def to(self, device):
        """
        Move the model to the specified device.

        :param device: The device to move the model to.
        """
        super().to(device)
        self.weight = self.weight.to(device)

        return self

    @property
    def mask(self):
        return torch.sigmoid(self.weight)

    @property
    def entropy(self):

        probs = torch.nn.functional.softmax(self.mask, -1).view(
            -1,
        )

        return torch.distributions.Categorical(probs).entropy()

    @property
    def budget_loss(self):
        return torch.relu(self.budget - self.mask.sum())
