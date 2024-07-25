import torch
import math
import copy
import time


class PathSGD:
    def __init__(self, model, input_dim):
        """
        Source: https://github.com/bneyshabur/path-sgd/blob/master/path_sgd.py
        :param model:
        :param input_dim:
        """
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.input_dim = input_dim
        self.state = model.state_dict()
        self.params = model.parameters()
        self.path_state = self.model.state_dict()
        self.path_params = self.model.named_parameters()
        self.ratio = None
        self.device = model.hparams.device

    # compute the scaling factors for each parameters. Path-SGD updates can then be calculated by element-wise devision
    # of gradient values by these scaling factors
    def compute_scaling(self):
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    param.pow_(2)
        data_ones = torch.ones((2, self.input_dim)).to(self.device)
        path_norm = 0.5 * self.model(data_ones).sum()
        path_norm.backward()

    # calculating the ratio of norm of gradient to the norm of Path-SGD updates. This ratio will be used to adjust the
    # the scaling of the Path-SGD update
    def compute_ratio(self):
        sgd_norm = 0
        pathsgd_norm = 0
        path_params = iter(self.model.parameters())
        for param in self.params:
            if param.requires_grad:
                sgd_norm += param.grad.norm()
                pathsgd_norm += (
                    param.grad.div(next(path_params).grad + 1e-8).norm() ** 2
                )
        self.ratio = (sgd_norm / pathsgd_norm) ** 0.5

        print(f"Ratio: {self.ratio}")

    # This functions modify the gradient of paramters of the model and set it to Path-SGD update
    def update_grad(self):
        self.model.load_state_dict(self.state)
        self.compute_scaling()
        path_params = iter(self.model.parameters())
        with torch.no_grad():
            if self.ratio is None:
                self.compute_ratio()
            for param in self.params:
                if param.requires_grad:
                    param.grad.div_(next(path_params).grad + 1e-8).mul_(self.ratio)
