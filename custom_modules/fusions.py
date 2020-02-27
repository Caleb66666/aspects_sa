# @File: fusions
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/26 14:50:00


import torch
from torch import nn


class BasicSfu(nn.Module):
    def __init__(self, input_size, fusion_size):
        super(BasicSfu, self).__init__()
        self.linear_r = nn.Linear(input_size + fusion_size, input_size)
        self.linear_g = nn.Linear(input_size + fusion_size, input_size)

    def forward(self, x, y):
        r_f = torch.cat([x, y], dim=2)
        r = torch.tanh(self.linear_r(r_f))
        g = torch.sigmoid(self.linear_g(r_f))
        o = g * r + (1 - g) * x
        return o


class SfuCombiner(nn.Module):
    def __init__(self, input_size, fusion_size, dropout=None, training=True):
        super(SfuCombiner, self).__init__()
        self.linear_r = nn.Linear((input_size + fusion_size) * 2, input_size)
        self.linear_g = nn.Linear((input_size + fusion_size) * 2, input_size)
        self.dropout = dropout
        self.training = training

    def forward(self, x, y):
        r_f = torch.cat([x, y, x * y, x - y], dim=2)
        if self.dropout is not None:
            r_f = nn.functional.dropout(r_f, p=self.dropout, training=self.training)
        r = torch.tanh(self.linear_r(r_f))
        g = torch.sigmoid(self.linear_g(r_f))
        o = g * r + (1 - g) * x
        return o


class NoneLinearGate(nn.Module):
    def __init__(self):
        super(NoneLinearGate, self).__init__()

    def forward(self, raw_x, fusion_x):
        gate_x = torch.sigmoid(raw_x)
        return fusion_x * gate_x


class Highway(nn.Module):
    def __init__(self, input_size, num_layers, identity=True, activate_f=None):
        super(Highway, self).__init__()
        self.identity = identity

        self.nonlinear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])
        if not self.identity:
            self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])
        self.activate = activate_f or nn.ReLU()

    def forward(self, x):
        """
        Args:
            x  (Tensor): [batch_size, size]
        Return:
            x' (Tensor): [batch_size, size]

        Applies σ(x) ⨀ f(G(x)) + (1 - σ(x)) ⨀ Q(x)

        linear: Q (affine transformation) or x (Identity)
        nonlinear: f (non-linear tranformation) with G (affine transformation)
        gate: σ(x) (affine transformation) with sigmoid
        ⨀: element-wise multiplication
        """
        if self.identity:
            for nonlinear, gate in zip(self.nonlinear, self.gate):
                gate = torch.sigmoid(gate(x))
                x = gate * self.activate(nonlinear(x)) + (1 - gate) * x
        else:
            for nonlinear, linear, gate in zip(self.nonlinear, self.linear, self.gate):
                gate = torch.sigmoid(gate(x))
                x = gate * self.activate(nonlinear(x)) + (1 - gate) * linear(x)

        return x


class Concat(nn.Module):
    def forward(self, x, y, dim=-1):
        return torch.cat([x, y], dim=dim)
