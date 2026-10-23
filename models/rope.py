import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from argparse import Namespace as ArgParseNamespace

def precompute_rope(cfg: ArgParseNamespace):
    """
    This function computes the sin and cos variants of RoPE, which is similar to HuggingFace's implementation.
    """
    E = cfg.model.embed_size
    Nh = cfg.model.num_heads
    H = E // Nh
    assert H % 2 == 0, "Head dim must be even!"

    theta = cfg.model.rope_theta
    seq_len = cfg.model.seq_len

    exp = torch.arange(0, H, 2).float() / H
    inv_freq = 1.0 / (theta ** exp)
    positions = torch.arange(seq_len) # this is m from m * \theta

    # this is an outer product
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0) # (seq_len, H // 2)

    # here we duplicate to the right: [a, b] becomes [a, b, a, b] and the shape is (seq_len, H)
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def apply_rope(x, cos, sin):
    B, Nh, T, H = x.shape
    assert H % 2 == 0, "Head dim must be even!"

    x1 = x[..., :(H // 2)] # first half
    x2 = x[..., (H // 2):] # second half

    # shape (1, 1, T, H)
    cos = cos[:T, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:T, :].unsqueeze(0).unsqueeze(0)

    # apply the rotary transformation
    rot = torch.cat((-x2, x1), dim=-1) # coeff of sine
    x_rot = (x * cos) + (rot * sin)

    return x_rot.to(dtype=x.dtype)