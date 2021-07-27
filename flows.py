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
    def __init__(self, num_inputs, num_hidden, learnable:bool=False, transform:callable=None):
        super().__init__()

        self.learnable = learnable
        self.transform = transform if transform is not None else lambda x: torch.clamp(torch.tril(x), 0.0, 1.0)

        self.input_mask_fix= flows.get_mask(num_inputs, num_hidden, num_inputs, mask_type='input')
        self.hidden_mask_fix = flows.get_mask(num_hidden, num_hidden, num_inputs)
        self.output_mask_fix = flows.get_mask(num_hidden, num_inputs, num_inputs, mask_type='output')

        self.num_ones = self.input_mask_fix.bool().sum() + self.hidden_mask_fix.bool().sum() + self.output_mask_fix.bool().sum()

        self.input = nn.Parameter(torch.ones_like(self.input_mask_fix)*self.input_mask_fix)
        self.hidden = nn.Parameter(torch.ones_like(self.hidden_mask_fix)*self.hidden_mask_fix)
        self.output= nn.Parameter(torch.ones_like(self.output_mask_fix)*self.output_mask_fix)

    def to(self, device):
        self.input.to(device)
        self.hidden.to(device)
        self.output.to(device)

        self.input_mask_fix= self.input_mask_fix.to(device)
        self.hidden_mask_fix = self.hidden_mask_fix.to(device)
        self.output_mask_fix = self.output_mask_fix.to(device)


        return self

    def mask(self):
        return self.output.data@self.hidden.data@self.input.data/self.num_ones

    def input_mask(self):
        return self.input_mask_fix if self.learnable is False else self.transform(self.input)*self.input_mask_fix


    def hidden_mask(self):
        return self.hidden_mask_fix if self.learnable is False else self.transform(self.hidden)*self.hidden_mask_fix

    def output_mask(self):
        return self.output_mask_fix if self.learnable is False else self.transform(self.output)*self.output_mask_fix

        
    def inject_structure(self, adj_mat, inject_structure=False):

        if inject_structure is True:
            # create an adjacency matrix with a full diagonal but different global structure than the original
            # the injected matrix is a lower triangular matrix and has at least one 0 element where the original matrix has a 1 element
            while torch.equal( (tmp_out:= torch.bernoulli(.5*torch.ones_like(self.output_mask_fix))*self.output_mask_fix)
                            @
                            (tmp_hid:= torch.bernoulli(.5*torch.ones_like(self.hidden_mask_fix))*self.hidden_mask_fix)
                            @
                            (tmp_in := torch.bernoulli(.5*torch.ones_like(self.input_mask_fix))*self.input_mask_fix), 
                            torch.tril(adj_mat, -1)) is True or ((tmp_out@tmp_hid@tmp_in)*adj_mat.bool()).sum() == adj_mat.bool().sum():
                pass

            
            

            self.input = nn.Parameter(tmp_in)
            self.hidden = nn.Parameter(tmp_hid)
            self.output = nn.Parameter(tmp_out)
            self.transform = lambda x: torch.clamp(x, 0.0, 1.0)

            print(f"Injected structure with weight: {self.mask()}")


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


class LearnableMaskedLinear(nn.Module):
    def __init__(self,in_features,out_features,learnable_mask,cond_in_features=None,bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=False)

        self.learnable_mask = learnable_mask

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.learnable_mask(),
                          self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output

class SplitMADESubnet(nn.Module):
    """
    A subnetwork of the split MADE
    """
    def __init__(self, num_inputs, num_hidden, num_components, confidence, activation, num_cond_inputs=None):
        super().__init__() 

        self.num_components = num_components

        self.confidence = confidence
        self.activation = activation

        self.input = LearnableMaskedLinear(num_inputs, num_hidden, self.confidence.input_mask, num_cond_inputs)
        self.hidden = LearnableMaskedLinear(num_hidden, num_hidden, self.confidence.hidden_mask)

        if self.num_components is None:
            self.output = LearnableMaskedLinear(num_hidden, num_inputs, self.confidence.output_mask)
        else:
            self.output = nn.ModuleList(
                [LearnableMaskedLinear(num_hidden, num_inputs, self.confidence.output_mask) for _ in range(num_components)])
    
    def forward(self, inputs, cond_inputs=None):

        h1 = self.activation(self.input(inputs, cond_inputs))
        h2 = self.activation(self.hidden(h1))

        if self.num_components is None:
            h3 = self.output(h2, )
        else:
            h3 = torch.stack([comp(h2, ) for comp in self.output], axis=0).mean(
                axis=0)
        return h3


class MaskMADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self, num_inputs, num_hidden, confidence,
                 num_cond_inputs=None, act=F.relu, pre_exp_tanh=False, num_components=1):
        super().__init__()


        self.pre_exp_tanh = pre_exp_tanh

        self.s_net = SplitMADESubnet(num_inputs, num_hidden, num_components, confidence,act, num_cond_inputs)
        self.t_net = SplitMADESubnet(num_inputs, num_hidden, num_components, confidence,act, num_cond_inputs)

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
        m = self.s_net(inputs, cond_inputs)
        a = self.t_net(inputs, cond_inputs)


        if self.pre_exp_tanh:
            a = torch.tanh(a)

        return a, m


class MaskMAF(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_blocks, act, use_reverse, use_batch_norm, learnable=False,
                 num_components=1, num_cond_inputs=None):
        super().__init__()

        self.use_reverse = use_reverse
        self.use_batch_norm = use_batch_norm
        self.num_blocks = num_blocks
        self.num_components = num_components

        self.confidence = EdgeConfidenceLayer(num_inputs, num_hidden, learnable)

        modules = []
        for i in range(self.num_blocks):
            modules += [
                MaskMADE(num_inputs, num_hidden, self.confidence,
                         num_cond_inputs, act=act, num_components=None if i != 0 else self.num_components)]

            if self.use_batch_norm is True:
                modules+=[flows.BatchNormFlow(num_inputs, momentum=0.1)]

            if self.use_reverse is True:
                modules += [flows.Reverse(num_inputs)]

        self.model = flows.FlowSequential(*modules)

    def forward(self, inputs, cond_inputs=None):
        # set_trace()
        # print(self.attention.attention.weight)
        outputs, logdets = self.model(inputs, cond_inputs)
        return outputs
