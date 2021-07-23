import importlib

flows = importlib.import_module("pytorch-flows.flows")
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace


# a big part of the code from: https://github.com/ikostrikov/pytorch-flows

class EdgeConfidenceLayer(nn.Module):
    """
    Calculates the confidence of each edge in the causal graph
    """
    def __init__(self, num_dim, transform:callable=None):
        super().__init__()
        self.weight = nn.Parameter(torch.tril(torch.ones(num_dim, num_dim),-1))

        self.transform = transform if transform is not None else lambda x: torch.clamp(torch.tril(x), 0.0, 1.0)



    def forward(self, x):
        return x@self.transform(self.weight)

    def inject_structure(self, adj_mat, inject_structure=False):

        if inject_structure is True:
            # create an adjacency matrix with a full diagonal but different global structure than the original
            # the injected matrix is a lower triangular matrix and has at least one 0 element where the original matrix has a 1 element
            while torch.equal(tmp := torch.clamp(torch.eye(adj_mat.shape[0]).to(adj_mat.device)+torch.tril(torch.bernoulli(.5*torch.ones_like(self.weight)), -1),0.,1.), adj_mat.bool() ) is True or tmp.sum() == 0 or (tmp*adj_mat.bool()).sum() == adj_mat.bool().sum():
                pass

            self.weight = nn.Parameter(torch.tril(tmp,-1), requires_grad=False)
            self.transform = lambda x: torch.clamp(x, 0.0, 1.0)

            print(f"Injected structure with weight: {self.weight}")


class AttentionNet(nn.Module):

    def __init__(self, attention_size, bias=False, transform=lambda x: x):
        super().__init__()

        self.transform = transform
        self.attention_size = attention_size

        self.attention = nn.Linear(self.attention_size, self.attention_size, bias=bias)

        self.tranform = transform
        self.attention.weight = nn.Parameter(self.transform(self.attention.weight))

    def forward(self, target, attn=None):
        # set_trace()
        return target * F.softmax(F.linear(target if attn is None else attn,self.transform(self.attention.weight),self.attention.bias), dim=-1)


class MaskNet(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()

        self.mask = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        return F.softmax(self.mask(input), dim=-1)



class MaskMADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self, num_inputs, num_hidden, confidence,
                 num_cond_inputs=None, act=F.relu, pre_exp_tanh=False, num_components=1):
        super().__init__()

        self.confidence = confidence
        self.num_components = num_components

        
        self.activation = act

        input_mask = flows.get_mask(num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = flows.get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = flows.get_mask(num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.input = flows.MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)
        self.hidden = flows.MaskedLinear(num_hidden, num_hidden, hidden_mask)

        if self.num_components is None:
            self.output = flows.MaskedLinear(num_hidden, num_inputs * 2, output_mask)
        else:
            self.output = nn.ModuleList(
                [flows.MaskedLinear(num_hidden, num_inputs * 2, output_mask) for _ in range(num_components)])

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
        # todo: attention is not invertible
        u = (inputs - self.confidence(m)) * torch.exp(-self.confidence(a))

        return u

    def conditioner_pass(self, inputs, cond_inputs):
        h1 = self.activation(self.input(inputs, cond_inputs))
        h2 = self.activation(self.hidden(h1))

        if self.num_components is None:
            h3 = self.output(h2, )
        else:
            h3 = torch.stack([comp(h2, ) for comp in self.output], axis=0).mean(
                axis=0)
        m, a = h3.chunk(2, 1)

        return a, m


class MaskMAF(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_blocks, act, use_reverse, num_components=1, num_cond_inputs=None):
        super().__init__()

        self.use_reverse = use_reverse
        self.num_blocks = num_blocks
        self.num_components = num_components

        self.confidence = EdgeConfidenceLayer(num_inputs)

        modules = []
        for i in range(self.num_blocks):
            modules += [
                MaskMADE(num_inputs, num_hidden, self.confidence,
                         num_cond_inputs, act=act, num_components=None if i != 0 else self.num_components),
                flows.BatchNormFlow(num_inputs, momentum=0.1)]

            if self.use_reverse is True:
                modules += [flows.Reverse(num_inputs)]

        self.model = flows.FlowSequential(*modules)

    def forward(self, inputs, cond_inputs=None):
        # set_trace()
        # print(self.attention.attention.weight)
        outputs, logdets = self.model(inputs, cond_inputs)
        return outputs
