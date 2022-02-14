import itertools
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import care_nl_ica.cl_ica.layers as ls
from care_nl_ica.models.sinkhorn import SinkhornNet
from care_nl_ica.models.sparsity import SparseBudgetNet

FeatureList = List[int]


class LinearSEM(nn.Module):
    def __init__(self, num_vars: int, permute: bool = False, variant: int = -1, force_chain: bool = False, force_uniform: bool = False):
        super().__init__()

        self.variant = variant
        self.num_vars = num_vars

        # weight init
        self.weight = nn.Parameter(torch.tril(nn.Linear(num_vars, num_vars).weight))
        if force_uniform is True:
            print('---------Forcing uniform weights---------')
            self.weight = nn.Parameter(torch.tril(torch.ones(num_vars, num_vars)))
        print(f'{self.weight=}')

        self.mask = (torch.tril(torch.bernoulli(0.65 * torch.ones_like(self.weight)), 1) + torch.eye(
            num_vars)).bool().float()


        # construct a chain
        if force_chain is True:
            print("-------Forcing chain-------")
            self.mask = torch.tril(torch.ones_like(self.weight))

            zeros_in_chain = torch.tril(torch.ones_like(self.weight), -2)
            self.mask[zeros_in_chain==1] = 0


        self.mask.requires_grad = False
        print(f"{self.mask=}")

        self._setup_permutation(permute)

    def _setup_permutation(self, permute):

        if self.variant == -1:
            self.permute_indices = torch.randperm(self.num_vars)
        else:
            if self.variant < (fac := math.factorial(self.num_vars)):
                permutations = list(itertools.permutations(range(self.num_vars)))
                self.permute_indices = torch.tensor(permutations[self.variant])
            else:
                raise ValueError(f'{self.variant=} should be smaller than {fac}')

        self.permutation = (lambda x: x) if permute is False else (lambda x: x[:, self.permute_indices])

        print(f'Using permutation with indices {self.permute_indices}')
        print(f"{self.permutation_matrix=}")

    @property
    def permutation_matrix(self) -> torch.Tensor:
        m = torch.zeros_like(self.weight)
        m[list(range(self.num_vars)), self.permute_indices] = 1

        return m

    def forward(self, x):
        z = torch.zeros_like(x)
        w = torch.tril(self.weight * self.mask)

        for i in range(self.num_vars):
            z[:, i] = w[i, i] * x[:, i]

            if i != 0:
                z[:, i] = z[:, i] + z[:, :i] @ w[i, :i]

        return self.permutation(z)

    def to(self, device):
        """
        Move the model to the specified device.

        :param device: The device to move the model to.
        """
        super().to(device)
        self.weight = self.weight.to(device)
        self.mask = self.mask.to(device)

        return self


class NonLinearSEM(LinearSEM):
    def __init__(self, num_vars: int, permute: bool = False, variant=-1, force_chain: bool = False, force_uniform: bool = False):
        super().__init__(num_vars=num_vars, permute=permute, variant=variant, force_chain=force_chain, force_uniform=force_uniform)

        self.nonlin = [lambda x: x ** 3, lambda x: torch.tanh(x), lambda x: torch.sigmoid(x),
                       lambda x: torch.nn.functional.leaky_relu(x, .1), lambda x: x]

        # Nonlinearitites
        self.nonlin_names = ['cube', 'tanh', 'sigmoid', 'leaky_relu', 'identity']
        self.nonlin_selector = torch.randint(0, len(self.nonlin) - 1, (num_vars,))

        # print the selected nonlinearities
        for i in range(num_vars):
            print(f"{self.nonlin_names[self.nonlin_selector[i]]}")

    def forward(self, x):

        z = torch.zeros_like(x)
        w = torch.tril(self.weight * self.mask)

        for i, nonlin_idx in enumerate(self.nonlin_selector):
            if i != 0:
                z[:, i] = self.nonlin[nonlin_idx](w[i, i] * x[:, i] + z[:, :i] @ w[i, :i])
            else:
                z[:, i] = w[i, i] * self.nonlin[nonlin_idx](x[:, i])

        return self.permutation(z)



class ARMLP(nn.Module):
    def __init__(self, num_vars: int, transform: callable = None, residual: bool = False, num_weights: int = 5,
                 triangular=True, budget:bool=False):
        super().__init__()

        self.num_vars = num_vars
        self.residual = residual
        self.triangular = triangular
        self.budget = budget

        if self.budget is True:
            self.budget_net = SparseBudgetNet(self.num_vars)

        self.weight = nn.ParameterList([
                                        nn.Parameter(
                                                        torch.tril(nn.Linear(num_vars, num_vars).weight, 0 if self.residual is False else -1)
                                                        if self.triangular is True else nn.Linear(num_vars, num_vars, bias=False).weight
                                                     )

                                        for _ in range(num_weights)
        ])
        if self.residual is True and self.triangular is True:
            self.scaling = nn.Parameter(torch.ones(self.num_vars), requires_grad=True)

        # structure injection
        self.transform = transform if transform is not None else lambda w: w
        self.struct_mask = torch.ones_like(self.weight[0], requires_grad=False)

    @property
    def assembled_weight(self):
        w = torch.ones_like(self.weight[0])
        for i in range(len(self.weight)):
            w *= self.weight[i]

        w = w if (self.residual is False or self.triangular is False) else w + torch.diag(self.scaling)

        assembled = w if self.triangular is False else torch.tril(w)

        if self.budget is True:
            assembled = assembled*self.budget_net.mask
        return assembled

    def forward(self, x):
        return self.transform(self.assembled_weight) @ x

    def to(self, device):
        """
        Move the model to the specified device.

        :param device: The device to move the model to.
        """
        super().to(device)
        self.weight = self.weight.to(device)

        if self.residual is True:
            self.scaling = self.scaling.to(device)

        if self.budget is True:
            self.budget_net = self.budget_net.to(device)
        return self

    def inject_structure(self, adj_mat, inject_structure=False):
        if inject_structure is True:
            # set structural mask
            self.struct_mask = (adj_mat.abs() > 0).float()
            self.struct_mask.requires_grad = False

            # set transform to include structural mask
            self.transform = lambda w: self.struct_mask * w

            print(f"Injected structure with weight: \n {self.struct_mask}")


class FeatureMLP(nn.Module):
    def __init__(self, num_vars: int, in_features: int, out_feature: int, bias: bool = True, force_identity: bool = False):
        super().__init__()
        self.num_vars = num_vars
        self.in_features = in_features
        self.out_feature = out_feature
        self.bias = bias

        # create MLPs
        self.mlps = nn.ModuleList(
            [nn.Linear(self.in_features, self.out_feature, self.bias) for _ in range(self.num_vars)])

        self.act = nn.ModuleList([nn.LeakyReLU() for _ in range(self.num_vars)])
        if force_identity is True:
            self.act = nn.ModuleList([nn.Identity() for _ in range(self.num_vars)])
            print("-----------------using identity activation-----------------")

    def forward(self, x):
        """

        :param x: tensor of size (batch x num_vars x in_features)
        :return:
        """

        if self.in_features == 1 and len(x.shape) == 2:
            x = torch.unsqueeze(x, 2)

        # the ith layer only gets the ith variable
        # reassemble into shape (batch_size, num_vars, out_features)
        return torch.stack([self.act[i](mlp(x[:, i, :])) for i, mlp in enumerate(self.mlps)], dim=1)


class PermutationNet(nn.Module):
    def __init__(self, num_vars):
        super().__init__()
        self.num_vars = num_vars
        self.weight = nn.Parameter(torch.randn(num_vars,))
        self.softmax = nn.Softmax(0)

        self.i = 0

    def forward(self,x):

        self.i += 1
        if self.i % 250 == 0:
            print(f"{self.weight=}")

        sorted, indices = self.weight.sort()
        perm = torch.zeros(self.num_vars, self.num_vars, device=self.weight.device)
        perm[list(range(self.num_vars)), indices] =sorted
        perm/=perm.sum(0)

        return perm@x

    def to(self, device):
        """
        Move the model to the specified device.

        :param device: The device to move the model to.
        """
        super().to(device)
        self.weight = self.weight.to(device)

        return self

class ARBottleneckNet(nn.Module):
    def __init__(self, num_vars: int, pre_layer_feats: FeatureList, post_layer_feats: FeatureList, bias: bool = True,
                 normalize: bool = False, residual: bool = False, permute=False, sinkhorn=False, triangular=True,
                 budget:bool=False):
        super().__init__()
        self.num_vars = num_vars
        self.pre_layer_feats = pre_layer_feats
        self.post_layer_feats = post_layer_feats
        self.bias = bias


        self._init_feature_layers()

        self.ar_bottleneck = ARMLP(self.num_vars, residual=residual, triangular=triangular, budget=budget)

        self.scaling = lambda x: x if normalize is False else ls.SoftclipLayer(self.num_vars, 1, True)

        self.sinkhorn = SinkhornNet(self.num_vars, 5, 1e-3)
        self.perm_net = PermutationNet(self.num_vars)

        self.inv_permutation = torch.arange(self.num_vars)

        self.permutation = (lambda x: x) if (permute is False or sinkhorn is False) else (lambda x: self.sinkhorn(x))

    def _layer_generator(self, features: FeatureList):
        return nn.Sequential(*[FeatureMLP(self.num_vars, features[idx], features[idx + 1], self.bias) for idx in
                               range(len(features) - 1)])

    def _init_feature_layers(self):
        """
        Initialzies the feature transform layers before and after the bottleneck.

        :return:
        """
        # check argument validity
        if not len(self.pre_layer_feats) and not len(self.post_layer_feats):
            raise ValueError(f"No pre- and post-layer specified!")

        self._init_pre_layers()
        self._init_post_layers()

    def _init_pre_layers(self):
        """
        Initialzies the feature transform layers before the bottleneck.

        :return:
        """
        if len(self.pre_layer_feats):
            # check feature values at the "interface"
            # input (coming from the outer world) has num_features=1
            if (first_feat := self.pre_layer_feats[0]) != 1:
                raise ValueError(f"First feature size should be 1, got {first_feat}!")

            # create layers with ReLU activations
            self.pre_layers = self._layer_generator(self.pre_layer_feats)

    def _init_post_layers(self):
        """
        Initialzies the feature transform layers after the bottleneck.

        :return:
        """
        if len(self.post_layer_feats):

            # check feature values at the "interface"
            # output has num_features=1
            if (last_feat := self.post_layer_feats[-1]) != 1:
                raise ValueError(f"Last feature size should be 1, got {last_feat}!")

            # create layers with ReLU activations
            self.post_layers = self._layer_generator(self.post_layer_feats)

    def forward(self, x):
        return self.scaling(torch.squeeze(self.post_layers(self.ar_bottleneck(self.permutation(self.pre_layers(x))))))
        # return self.ar_bottleneck(x.T).T

    def to(self, device):
        """
        Moves the model to the specified device.

        :param device: device to move the model to
        :return: self
        """
        super().to(device)
        # move the model to the specified device
        self.pre_layers.to(device)
        self.post_layers.to(device)
        self.ar_bottleneck.to(device)
        self.sinkhorn.to(device)
        self.inv_permutation.to(device)
        self.perm_net.to(device)

        return self

    @property
    def bottleneck_l1_norm(self):
        return self.ar_bottleneck.assembled_weight.abs().l1_loss()




if __name__ == "__main__":
    NUM_DIM = 3
    L1 = 5e0
    s = SinkhornNet(NUM_DIM,20,3e-4)
    permute_indices = [1,2,0]

    permute_mat = torch.zeros(NUM_DIM,NUM_DIM)
    permute_mat[list(range(NUM_DIM)), permute_indices]=1

    causal2repr = torch.randn_like(permute_mat)

    # generate chain
    weight = torch.tril(torch.ones_like(permute_mat), -2)
    mask = torch.tril(torch.ones_like(weight))
    zeros_in_chain = torch.tril(torch.ones_like(weight), -2)
    mask[zeros_in_chain == 1] = 0

    # J

    class ReprNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.unmixing = torch.nn.Parameter(torch.tril(torch.randn_like(weight)), requires_grad=True)
            self.repr_weight = torch.nn.Parameter(torch.randn_like(weight), requires_grad=True)
        def forward(self, x):
            return torch.tril(self.unmixing)@self.repr_weight @x


    mixing = torch.tril(torch.randn(NUM_DIM, NUM_DIM)) * mask
    J_repr = causal2repr@mixing


    print(f"{J_repr=}")
    reprnet= ReprNet()
    optim = torch.optim.Adam(reprnet.parameters(), lr=1e-3)

    from metrics import frobenius_diagonality

    for i in range(12000):

        optim.zero_grad()

        diagonality = frobenius_diagonality(reprnet(causal2repr @ mixing))
        l1_loss = torch.tril(reprnet.unmixing).abs().mean()
        loss = diagonality + L1 * l1_loss

        if i % 250 == 0:
            print(f"{diagonality=}, {l1_loss=}")

        loss.backward()


        optim.step()


    print(f"{mixing=}")
    print(f"{reprnet.unmixing=}")
    print(f"{causal2repr.T.qr()=}")
    print(f"{reprnet.repr_weight.T.qr()=}")
