import inspect
import math
from types import LambdaType

import torch
import torch.nn as nn

from .utils import get_batcnnorm, get_dropout, get_activation


__all__ = [
    'LinearLayer', 'MLPLayer', 'FeatureSelector',
]


class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma, device):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(0.01*torch.randn(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size()) 
        self.sigma = sigma
        self.device = device
    
    def forward(self, prev_x):
        z = self.mu + self.sigma*self.noise.normal_()*self.training 
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x
    
    def hard_sigmoid(self, x):
        return torch.clamp(x+0.5, 0.0, 1.0)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) 

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self


class GatingNet(nn.Module):
    def __init__(self, input_dim, gating_net_hidden_dims, sigma,
                device, activation, batch_norm, dropout):
        super(GatingNet, self).__init__()
        #self.mu = torch.nn.Parameter(0.01*torch.randn(input_dim, ), requires_grad=True)
        self.net = MLPLayer(input_dim, input_dim, gating_net_hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation='tanh')
        #self.noise = torch.randn(self.mu.size()) 
        self.noise = torch.randn(input_dim) 
        self.sigma = sigma
        self.device = device

    def calc_mu(self, x):
        return self.net(x)
    
    def forward(self, prev_x):
        mu = self.calc_mu(prev_x)
        z = mu + self.sigma*self.noise.normal_()*self.training 
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x, mu
    
    def hard_sigmoid(self, x):
        return torch.clamp(x+0.5, 0.0, 1.0)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) 

    def _apply(self, fn):
        super(GatingNet, self)._apply(fn)
        self.noise = fn(self.noise)
        return self

class GatingLayer(nn.Module):
    '''To implement L1-based gating layer (so that we can compare L1 with L0(STG) in a fair way)
    '''
    def __init__(self, input_dim, device):
        super(GatingLayer, self).__init__()
        self.mu = torch.nn.Parameter(0.01*torch.randn(input_dim, ), requires_grad=True)
        self.device = device
    
    def forward(self, prev_x):
        new_x = prev_x * self.mu 
        return new_x
    
    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return torch.sum(torch.abs(x))


class LinearLayer(nn.Sequential):
    def __init__(self, in_features, out_features, batch_norm=None, dropout=None, bias=None, activation=None):
        if bias is None:
            bias = (batch_norm is None)

        modules = [nn.Linear(in_features, out_features, bias=bias)]
        if batch_norm is not None and batch_norm is not False:
            modules.append(get_batcnnorm(batch_norm, out_features, 1))
        if dropout is not None and dropout is not False:
            modules.append(get_dropout(dropout, 1))
        if activation is not None and activation is not False and activation != 'none':
            modules.append(get_activation(activation))
        super().__init__(*modules)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class LambdaModule(nn.Module):
    def __init__(self, lambda_function: LambdaType):
        super().__init__()
        import types
        assert isinstance(lambda_function, LambdaType)
        self.lambda_function = lambda_function

    def extra_repr(self) -> str:
        return inspect.getsource(self.lambda_function).replace('\n', '')

    def __repr__(self):
        return self.extra_repr()

    def forward(self, x):
        return self.lambda_function(x)


ViewLayer = LambdaModule(lambda x, shape_tuple: x.view(shape_tuple))
# double lambda, external for allowing dynamic passing of the split dim, and internal as the input for LambdaModule
# This splits a 2D batch x features vector into a 3D batch/split x split x features vector
BatchSplit = lambda recurrent_split_dim: LambdaModule(lambda x: x.view(x.shape[0] // recurrent_split_dim, -1, x.shape[-1]))
BatchFlatten = LambdaModule(lambda x: x.view(-1, x.shape[-1]))
# For Rnn Output
FirstOfTuple = LambdaModule(lambda x: x[0])


class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_norm=None, dropout=None, activation='relu',
                 flatten=True, recurrent_split_dim=None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = []
        elif type(hidden_dims) is int:
            hidden_dims = [hidden_dims]

        assert (recurrent_split_dim is None) or (recurrent_split_dim > 1), "recurrent_split_dim must be None or int > 1"
        self.recurrent = recurrent_split_dim is not None

        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(output_dim)
        modules = []

        if self.recurrent:
            modules.append(BatchSplit(recurrent_split_dim))

        nr_hiddens = len(hidden_dims)
        for i in range(nr_hiddens):
            if self.recurrent:
                layer = nn.Sequential(nn.GRU(dims[i], dims[i+1], dropout=0 if dropout is None else dropout, batch_first=True), FirstOfTuple)
            else:
                layer = LinearLayer(dims[i], dims[i+1], batch_norm=batch_norm, dropout=dropout, activation=activation)
            modules.append(layer)
        layer = nn.Linear(dims[-2], dims[-1], bias=True)
        modules.append(layer)
        if self.recurrent:
            modules.append(BatchFlatten)
        self.mlp = nn.Sequential(*modules)
        self.flatten = flatten

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, input):
        if self.flatten:
            input = input.view(input.size(0), -1)
        return self.mlp(input)


class MLPLayerEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dims, batch_norm=None, dropout=None, activation='relu',
                 flatten=True, recurrent_split_dim=None):
        super().__init__()

        assert (recurrent_split_dim is None) or (recurrent_split_dim > 1), "recurrent_split_dim must be None or int > 1"
        self.recurrent = recurrent_split_dim is not None

        if hidden_dims is None:
            hidden_dims = []
        elif type(hidden_dims) is int:
            hidden_dims = [hidden_dims]

        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(encoding_dim)
        nr_hiddens = len(hidden_dims)

        modules = []
        if self.recurrent:
            modules.append(BatchSplit(recurrent_split_dim))
        for i in range(nr_hiddens):
            if self.recurrent:
                layer = nn.Sequential(nn.GRU(dims[i], dims[i+1], dropout=0 if dropout is None else dropout, batch_first=True), FirstOfTuple)
            else:
                layer = LinearLayer(dims[i], dims[i+1], batch_norm=batch_norm, dropout=dropout, activation=activation)
            modules.append(layer)
        layer = nn.Linear(dims[-2], dims[-1], bias=True)
        modules.append(layer)
        self.encoder = nn.Sequential(*modules)

        modules = []
        for i in range(nr_hiddens + 1, 1, -1):
            if self.recurrent:
                layer = nn.Sequential(nn.GRU(dims[i], dims[i-1], dropout=0 if dropout is None else dropout, batch_first=True), FirstOfTuple)
            else:
                layer = LinearLayer(dims[i], dims[i-1], batch_norm=batch_norm, dropout=dropout, activation=activation)
            modules.append(layer)
        if self.recurrent:
            layer = nn.Sequential(nn.GRU(dims[1], dims[0], batch_first=True), FirstOfTuple)
            modules.append(layer)
            modules.append(BatchFlatten)
        else:
            layer = nn.Linear(dims[1], dims[0], bias=True)
            modules.append(layer)

        self.decoder = nn.Sequential(*modules)

        self.flatten = flatten

    def encode(self, input):
        if self.flatten:
            input = input.view(input.size(0), -1)
        return self.encoder(input)

    def decode(self, encoding):
        return self.decoder(encoding)

    def forward(self, input):
        if self.flatten:
            input = input.view(input.size(0), -1)

        encoding = self.encode(input)
        out = self.decode(encoding)

        return out
