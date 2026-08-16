import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .base import BaseQuantizer


class NoiseQuantizer(BaseQuantizer):
    def __init__(self, bits=4, group_size=-1):
        super().__init__(bits)
        self.noise_scale = 2 ** (-bits)
        self.group_size = group_size
    

    @torch.compiler.disable()
    def forward(self, x):
        orig_shape = x.shape
        
        if self.group_size > 0:
            x = x.view(-1, self.group_size)
        
        std = torch.std(x, correction=0, dim=-1, keepdim=True)
        x_noise = x + torch.randn_like(x) * std * self.noise_scale
        return x_noise.reshape(*orig_shape)
