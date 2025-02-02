import math
import torch
import torch.nn as nn


class PosEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.shape[-2]]


class SpaEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.linear = nn.Linear(c_in, d_model)

    def forward(self, x):
        return self.linear(x)


class TemporalEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.te_scale = nn.Linear(c_in, d_model // 2)
        self.te_periodic = nn.Linear(c_in, d_model // 2)

    def forward(self, t):
        out1 = self.te_scale(t)
        out2 = torch.sin(self.te_periodic(t))
        return torch.cat([out1, out2], -1)
