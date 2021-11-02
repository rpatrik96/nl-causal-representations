from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

FeatureList = List[int]



class LinearSEM(nn.Module):
    def __init__(self, num_vars: int):
        super().__init__()


        self.num_vars = num_vars

        self.weight = nn.Parameter(torch.tril(nn.Linear(num_vars, num_vars).weight))

    def forward(self, x):
        from pdb import set_trace
        # set_trace()
        return x @ torch.tril(self.weight).T 

    def to(self, device):
        """
        Move the model to the specified device.

        :param device: The device to move the model to.
        """
        super().to(device)
        self.weight = self.weight.to(device)

        return self




class ARMLP(nn.Module):
    def __init__(self, num_vars: int, transform:callable=None):
        super().__init__()
        from pdb import set_trace
        # set_trace()

        self.weight = nn.Parameter(torch.tril(nn.Linear(num_vars, num_vars).weight))

        # structure injection
        self.transform = transform if transform is not None else lambda x: x
        self.struct_mask = torch.ones_like(self.weight, requires_grad=False)

    def forward(self, x):
        from pdb import set_trace
        # set_trace()
        return self.transform(torch.tril(self.weight) @ x)

    def to(self, device):
        """
        Move the model to the specified device.

        :param device: The device to move the model to.
        """
        super().to(device)
        self.weight = self.weight.to(device)

        return self

    def inject_structure(self, adj_mat, inject_structure=False):
        if inject_structure is True:

            # set structural mask
            self.struct_mask = adj_mat > 0
            
            # set transform to include structural mask
            self.transform = lambda x: self.struct_mask @ x


            print(f"Injected structure with weight: {self.struct_mask}")


class FeatureMLP(nn.Module):
    def __init__(self, num_vars: int, in_features: int, out_feature: int, bias: bool = True):
        super().__init__()
        self.num_vars = num_vars
        self.in_features = in_features
        self.out_feature = out_feature
        self.bias = bias

        # create MLPs
        self.mlps = nn.ModuleList(
            [nn.Linear(self.in_features, self.out_feature, self.bias) for _ in range(self.num_vars)])

        self.act = nn.ModuleList([nn.LeakyReLU() for _ in range(self.num_vars)])

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


class ARBottleneckNet(nn.Module):
    def __init__(self, num_vars: int, pre_layer_feats: FeatureList, post_layer_feats: FeatureList, bias: bool = True):
        super().__init__()
        self.num_vars = num_vars
        self.pre_layer_feats = pre_layer_feats
        self.post_layer_feats = post_layer_feats
        self.bias = bias

        self._init_feature_layers()

        self.ar_bottleneck = ARMLP(self.num_vars)

    def _layer_generator(self, features: FeatureList):
        return  nn.Sequential(*[FeatureMLP(self.num_vars, features[idx], features[idx+1]) for idx in range(len(features)-1)])

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
            # output (going into the bottleneck) has num_features=self.vars
            if (last_feat := self.pre_layer_feats[-1]) != self.num_vars:
                raise ValueError(f"Last feature size should be {self.num_vars}, got {last_feat}!")

            # create layers with ReLU activations
            self.pre_layers = self._layer_generator(self.pre_layer_feats)

    def _init_post_layers(self):
        """
        Initialzies the feature transform layers after the bottleneck.

        :return:
        """
        if len(self.post_layer_feats):

            # check feature values at the "interface"
            # input (coming from the bottleneck) has num_features=self.vars
            if (first_feat := self.post_layer_feats[0]) != self.num_vars:
                raise ValueError(f"First feature size should be {self.num_vars}, got {first_feat}!")
            # output has num_features=1
            if (last_feat := self.post_layer_feats[-1]) != 1:
                raise ValueError(f"Last feature size should be 1, got {last_feat}!")

            # create layers with ReLU activations
            self.post_layers = self._layer_generator(self.post_layer_feats)

    def forward(self, x):
        from pdb import set_trace
        # set_trace()
        return torch.squeeze(self.post_layers(self.ar_bottleneck(self.pre_layers(x))))

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

        return self


    @property
    def bottleneck_l1_norm(self):
        return self.ar_bottleneck.weight.abs().sum()

