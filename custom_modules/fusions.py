# @File: fusions
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/26 14:50:00


import torch
from torch import nn


class SFU(nn.Module):
    def __init__(self, input_size, fusion_size):
        super(SFU, self).__init__()
        self.linear_r = nn.Linear(input_size + fusion_size, input_size)
        self.linear_g = nn.Linear(input_size + fusion_size, input_size)

    def forward(self, raw_x, fusion_x):
        r_f = torch.cat([raw_x, fusion_x], dim=2)
        r = torch.tanh(self.linear_r(r_f))
        g = torch.sigmoid(self.linear_g(r_f))
        o = g * r + (1 - g) * raw_x
        return o


class NoneLinearGate(nn.Module):
    def __init__(self):
        super(NoneLinearGate, self).__init__()

    def forward(self, raw_x, fusion_x):
        gate_x = torch.sigmoid(raw_x)
        return fusion_x * gate_x
