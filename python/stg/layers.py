import inspect
import math
from types import LambdaType
from random import randint

import torch
import torch.nn as nn

from utils import get_batcnnorm, get_dropout, get_activation


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
    def __init__(self, input_dim, gating_net_hidden_dims, sigma, device, activation, batch_norm, dropout, pooling=False,
                 gumble=False, fixed_gates=None, mixed_weights=None):
        super(GatingNet, self).__init__()
        #self.mu = torch.nn.Parameter(0.01*torch.randn(input_dim, ), requires_grad=True)
        self.net = MLPLayer(input_dim, input_dim, gating_net_hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation='tanh')
        #self.noise = torch.randn(self.mu.size()) 
        self.noise = torch.randn(input_dim) 
        self.sigma = sigma
        self.device = device
        self.pooling = None
        if pooling:
            self.pooling = lambda x: x.mean(-1).mean(-1)
        self.fixed_gates = fixed_gates
        self.mixed_weights = mixed_weights
        self.gumble = gumble

    def calc_mu(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        return self.net(x)
    
    def forward(self, prev_x, include_losses=False, fixed_gates=None):
        if fixed_gates is not None or self.fixed_gates is not None:
            stochastic_gate = fixed_gates if fixed_gates is not None else self.fixed_gates
            # Fix later to mu = None , requires fixing inference not to compute loss
            mu = self.calc_mu(prev_x)
            if self.mixed_weights is not None:
                z = mu + self.sigma * self.noise.normal_() * self.training
                computed_gate = self.hard_sigmoid(z)
                (p1, p2) = self.mixed_weights
                # geometric average of the gates
                # stochastic_gate = (stochastic_gate ** p1 * computed_gate ** p2) ** (1 / (p1 + p2))
                stochastic_gate = (stochastic_gate * p1 + computed_gate * p2) / (p1 + p2)
        else:
            mu = self.calc_mu(prev_x)
            z = mu + self.sigma*self.noise.normal_() * self.training
            stochastic_gate = self.hard_sigmoid(z)
        gate_stats = stochastic_gate.sum().item() / torch.numel(stochastic_gate)
        if self.pooling is not None:
            stochastic_gate = stochastic_gate.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, prev_x.size(2), prev_x.size(3))
        new_x = prev_x * stochastic_gate
        if include_losses:
            return new_x, mu, stochastic_gate, (self.regularizer_loss(mu), self.similarity_loss(mu), gate_stats)
        return new_x, mu, stochastic_gate, None
    
    def hard_sigmoid(self, x):
        if self.gumble:
            out = torch.sigmoid(x)
            if True: #self.hard_gating:
                # create 0.5 constants to compare sigmoid to
                half_padding = 0.5 * torch.ones_like(out, requires_grad=True)
                # combine together on the last axis
                padded = torch.cat([half_padding.unsqueeze(-1), out.unsqueeze(-1)], -1)
                # get a binary result which one is bigger
                max_ind = padded.argmax(-1).float()
                # trick to enable gradient
                gating = (max_ind - out).detach() + out
            return gating
        else:
            return torch.clamp(x+0.5, 0.0, 1.0)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def regularizer_loss(self, mu):
        return torch.mean(self.regularizer((mu + 0.5) / self.sigma))

    def similarity_loss(self, mu):
        mu = self.hard_sigmoid(mu)
        mu_T = mu.T

        mu_norm = torch.linalg.norm(mu, dim=1, keepdim=True)  # Size (n, 1).
        mu_T_norm = torch.linalg.norm(mu_T, dim=0, keepdim=True)  # Size (1, b).

        # Distance matrix of size (b, n).
        cosine_similarity = ((mu @ mu_T) / (mu_norm @ mu_T_norm + 1e-6)).T
        normalized = (cosine_similarity - torch.eye(mu.size(0), device=mu.device))

        adjustment_weights_mats = getattr(self, 'adjustment_weights_mats', None)
        # hacky_second_half_duplicate
        if adjustment_weights_mats is not None and adjustment_weights_mats[0].size(0) == normalized.size(0):
            r_ind = randint(0, len(adjustment_weights_mats)-1)
            adjustment_weights = adjustment_weights_mats[r_ind]
            normalized = normalized * adjustment_weights

        normalized = normalized.mean()
        return normalized

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
                 flatten=True, recurrent_split_dim=None, feature_selector: nn.Module = None):
        super().__init__()

        assert (recurrent_split_dim is None) or (recurrent_split_dim > 1), "recurrent_split_dim must be None or int > 1"
        self.recurrent = recurrent_split_dim is not None
        self.feature_selector = feature_selector
        self.last_mu = None

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
        if self.feature_selector is not None:
            encoding, mu = self.feature_selector(encoding)
            self.last_mu = mu
        out = self.decode(encoding)

        return out


def create_augmentation_weights_matrix(batch_size, repeat_ratio, extra_shift=0,multiplier=-1):
    assert (batch_size % repeat_ratio) == 0
    block = batch_size // repeat_ratio
    mat = torch.zeros(batch_size, batch_size)
    for i in range(1, repeat_ratio):
        shift = i*block + extra_shift
        aa = torch.roll(torch.eye(batch_size),shift,1) * multiplier
        aa[-shift:,:shift] = 0
        mat = mat + aa
    return mat + mat.T


def create_pos_neg_wieghts_matrices(batch_size, device):
    base_weights = create_augmentation_weights_matrix(batch_size,2)
    mats = []
    permutes = []
    half_batch = batch_size // 2
    for i in range(batch_size):
        permutes.append(torch.randperm(batch_size-2))
    for j in range(batch_size - 2):
        weights_mat = base_weights.clone()
        for i in range(batch_size):
            val_permute = int(permutes[i][j])
            # shift by 0, 1 or 2 depdending if before between or after main diagonal and -1 (their order changes in the top and buttom half)
            if i < half_batch:
                if val_permute >= i:
                    val_permute += 1
                if val_permute >= i + half_batch:
                    val_permute += 1
            else:
                if val_permute >= i - half_batch:
                    val_permute += 1
                if val_permute >= i:
                    val_permute += 1

            weights_mat[i, val_permute] = 1
        mats.append(weights_mat.to(device))
    return mats


if __name__ == '__main__':
    aaa = create_pos_neg_wieghts_matrices(8, "cpu")
    print(aaa[1])
    print(aaa[2])