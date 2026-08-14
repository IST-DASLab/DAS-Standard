import torch
from torch import nn as nn
from torch.nn import functional as F


class KarpathyLayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim: int, bias: bool, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=self.eps)


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * norm_x + self.bias


class RMSNorm(nn.Module):
    """
        Root Mean Square Layer Normalization: https://arxiv.org/pdf/1910.07467
    """
    def __init__(self, ndim: int, bias: bool=False, eps: float = 1e-5, qwen3_compatible: bool=False):
        super().__init__()
        self.ndim = ndim
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.weight = nn.Parameter(torch.ones(ndim, dtype=torch.float))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(dtype=torch.float32)

        var = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(var + self.eps) * self.weight

        if self.bias is not None:
            norm_x = norm_x + self.bias

        return norm_x.to(dtype=input_dtype.dtype)
