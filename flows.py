import importlib

flows = importlib.import_module("pytorch-flows.flows")
import torch
import torch.nn as nn
import torch.nn.functional as F


# a big part of the code from: https://github.com/ikostrikov/pytorch-flows


class MaskNet(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()

        self.mask = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        return F.softmax(self.mask(input), dim=-1)


class DoubleMaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, cond_in_features=None, bias=False):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=bias)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None, *, learnable_mask):
        output = F.linear(inputs, self.linear.weight * self.mask * learnable_mask,
                          self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output


class MaskMADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self, num_inputs, num_hidden, learnable_in_mask, learnable_hidden_mask, learnable_out_mask,
                 num_cond_inputs=None, act='relu', pre_exp_tanh=False, num_components=None):
        super().__init__()

        self.num_components = num_components

        self.learnable_in_mask = learnable_in_mask
        self.learnable_hidden_mask = learnable_hidden_mask
        self.learnable_out_mask = learnable_out_mask

        activations = {'relu': F.relu, 'sigmoid': F.sigmoid, 'tanh': F.tanh}
        self.activation = activations[act]

        input_mask = flows.get_mask(num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = flows.get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = flows.get_mask(num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.input = DoubleMaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)
        self.hidden = DoubleMaskedLinear(num_hidden, num_hidden, hidden_mask)

        if self.num_components is None:
            self.output = DoubleMaskedLinear(num_hidden, num_inputs * 2, output_mask)
        else:
            self.output = nn.ModuleList(
                [DoubleMaskedLinear(num_hidden, num_inputs * 2, output_mask) for _ in range(num_components)])

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            a, m = self.conditioner_pass(inputs, cond_inputs)

            u = self.transformer_fwd_pass(a, inputs, m)

            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                a, m = self.conditioner_pass(inputs, cond_inputs)

                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)

    def transformer_fwd_pass(self, a, inputs, m):
        u = (inputs - m) * torch.exp(-a)

        return u

    def conditioner_pass(self, inputs, cond_inputs):
        h1 = self.activation(self.input(inputs, cond_inputs, learnable_mask=self.learnable_in_mask))
        h2 = self.activation(self.hidden(h1, learnable_mask=self.learnable_hidden_mask))

        if self.num_components is None:
            h3 = self.output(h2, learnable_mask=self.learnable_out_mask)
        else:
            h3 = torch.stack([comp(h2, learnable_mask=self.learnable_out_mask) for comp in self.output], axis=0).mean(
                axis=0)
        m, a = h3.chunk(2, 1)

        return a, m


class MaskMAF(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_cond_inputs, num_blocks, num_components, act):
        super().__init__()

        self.num_blocks = num_blocks
        self.num_components = num_components

        input_mask = flows.get_mask(num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = flows.get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = flows.get_mask(num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.input_mask = MaskNet(*input_mask.shape, bias=False)
        self.hidden_mask = MaskNet(*hidden_mask.shape, bias=False)
        self.output_mask = MaskNet(*output_mask.shape, bias=False)

        modules = []
        for i in range(self.num_blocks):
            modules += [
                MaskMADE(num_inputs, num_hidden, input_mask, hidden_mask, output_mask, num_cond_inputs, act=act,
                         num_components=None if i != 0 else self.num_components),
                flows.BatchNormFlow(num_inputs),
                flows.Reverse(num_inputs)]

        self.model = flows.FlowSequential(*modules)

    def forward(self, inputs, cond_inputs=None):
        self.model(inputs, cond_inputs)

    @property
    def jacobian(self):
        return self.output_mask.weight.data @ self.hidden_mask.weight.data @ self.input_mask.weight.data


if __name__ == "__main__":
    num_inputs = 3
    num_hidden = 10
    num_outputs = 2
    num_blocks = 3
    num_components = 5
    batch_size = 32
    act = "relu"

    maf = MaskMAF(num_inputs, num_hidden, num_outputs, num_blocks, num_components, act)

    maf(torch.randn(batch_size, num_inputs))
