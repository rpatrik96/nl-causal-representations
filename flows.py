import importlib

flows = importlib.import_module("pytorch-flows.flows")
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# a big part of the code from: https://github.com/ikostrikov/pytorch-flows


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class AttentionNet(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.attention = nn.Linear(in_features, out_features)

    def forward(self, target, attn=None):
        return target * F.softmax(self.attention(target if attn is None else attn), dim=-1)


class DoubleMaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=False)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None, *, learnable_mask):
        output = F.linear(inputs, self.linear.weight * self.mask * learnable_mask,
                          self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output


class AttentionMADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self, num_inputs, num_hidden, learnable_in_mask, learnable_hid_mask, learnable_out_mask,
                 num_cond_inputs=None, act='relu', pre_exp_tanh=False):
        super().__init__()

        self.learnable_in_mask = learnable_in_mask
        self.learnable_hid_mask = learnable_hid_mask
        self.learnable_out_mask = learnable_out_mask

        activations = {'relu': F.relu, 'sigmoid': F.sigmoid, 'tanh': F.tanh}
        self.activation = activations[act]

        input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.input = DoubleMaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)
        self.hidden = DoubleMaskedLinear(num_hidden, num_hidden, hidden_mask)
        self.output = DoubleMaskedLinear(num_hidden, num_inputs * 2, output_mask)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h1 = self.activation(self.input(inputs, cond_inputs, learnable_mask=self.learnable_in_mask))
            h2 = self.activation(self.hidden(h1, learnable_mask=self.learnable_hid_mask))
            h3 = self.output(h2, learnable_mask=self.learnable_out_mask)

            m, a = h3.chunk(2, 1)

            u = (inputs - m) * torch.exp(-a)

            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.input(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)



class AttentionMAF(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_cond_inputs, num_blocks, act):
        super().__init__()

        self.num_blocks = num_blocks

        input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.input_mask = AttentionNet(*input_mask.shape)
        self.hidden_mask = AttentionNet(*hidden_mask.shape)
        self.output_mask = AttentionNet(*output_mask.shape)

        modules = []
        for _ in range(self.num_blocks):
            modules = [AttentionMADE(num_inputs, num_hidden, input_mask, hidden_mask, output_mask, num_cond_inputs, act=act),
                       flows.BatchNormFlow(num_inputs),
                       flows.Reverse(num_inputs)]

        self.model = flows.FlowSequential(*modules)


    def forward(self, inputs, cond_inputs=None):
        self.model(inputs, cond_inputs)


if __name__ == "__main__":

    num_inputs = 3
    num_hidden = 10
    num_outputs = 2
    num_blocks = 3
    batch_size = 32
    act = "relu"

    maf = AttentionMAF(num_inputs, num_hidden, num_outputs, num_blocks, act)

    maf(torch.randn(batch_size, num_inputs))
