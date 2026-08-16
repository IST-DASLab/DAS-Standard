import itertools
import math
from functools import cache

import torch
from torch import nn

from .base import BaseQuantizer


_E8P_CODESZ = 8


def get_norm12():
    # 29 elements of norm 12 in E8 + 1/4
    return torch.tensor([
        [3, 1, 1, 1, 3, 3, 3, 3],
        [1, 3, 1, 1, 3, 3, 3, 3],
        [1, 1, 3, 1, 3, 3, 3, 3],
        [1, 1, 1, 3, 3, 3, 3, 3],
        [3, 3, 3, 1, 3, 3, 1, 1],
        [3, 3, 3, 1, 3, 1, 3, 1],
        [3, 3, 3, 1, 1, 3, 3, 1],
        [3, 3, 3, 1, 3, 1, 1, 3],
        [3, 3, 3, 1, 1, 3, 1, 3],
        [3, 3, 3, 1, 1, 1, 3, 3],
        [3, 3, 1, 3, 3, 3, 1, 1],
        [3, 3, 1, 3, 3, 1, 3, 1],
        [3, 3, 1, 3, 1, 3, 3, 1],
        [3, 3, 1, 3, 3, 1, 1, 3],
        [3, 3, 1, 3, 1, 3, 1, 3],
        [3, 3, 1, 3, 1, 1, 3, 3],
        [3, 1, 3, 3, 3, 3, 1, 1],
        [3, 1, 3, 3, 3, 1, 3, 1],
        [3, 1, 3, 3, 1, 3, 3, 1],
        [3, 1, 3, 3, 3, 1, 1, 3],
        [3, 1, 3, 3, 1, 3, 1, 3],
        [1, 3, 3, 3, 1, 1, 3, 3],
        [1, 3, 3, 3, 3, 3, 1, 1],
        [1, 3, 3, 3, 3, 1, 3, 1],
        [1, 3, 3, 3, 1, 3, 3, 1],
        [1, 3, 3, 3, 3, 1, 1, 3],
        [1, 3, 3, 3, 1, 3, 1, 3],
        [1, 1, 3, 3, 1, 3, 3, 3],
        [3, 3, 1, 1, 3, 3, 3, 1],
    ]) / 2


def get_packed_abs_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * 8).float() + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1)**2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    norm12 = get_norm12()
    cba = torch.concat([d8abs, norm12], dim=0)
    cba = cba[:, [0, 2, 4, 6, 1, 3, 5, 7]]
    cba[:, 7] *= (1 - 2 * (cba.sum(1) % 2))
    cba = cba * 2 + 8
    cba = cba.to(torch.int32)
    acc = cba[:, 0]
    for i in range(7):
        acc = acc | (cba[:, (i + 1)] << ((i + 1) * 4))
    return acc


def get_abs_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * _E8P_CODESZ).float() + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1)**2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    norm12 = get_norm12()
    cba = torch.concat([d8abs, norm12], dim=0)
    return cba


def get_full_grid(packed_abs_grid):
    synth_codebook = torch.zeros(1 << 16, 8)
    parity_idx = []
    shuffle_map = [0, 4, 1, 5, 2, 6, 3, 7]
    for c in range(1 << 16):
        signs = c & 255
        abs = c >> 8
        parity = 0
        for i in range(8):
            parity = parity ^ ((signs >> i) & 1)
        signs = signs ^ parity
        abs_code = packed_abs_grid[abs].item()
        for i in range(8):
            ii = shuffle_map[i]
            synth_codebook[c, i] = (((abs_code >> (4 * ii)) & 15) - 8) * 0.5
            if ((signs >> ii) & 1):
                synth_codebook[c, i] *= -1
        if parity:
            synth_codebook[c, :] -= 0.25
            parity_idx.append(c)
        else:
            synth_codebook[c, :] += 0.25
    return synth_codebook, torch.arange(1 << 16), parity_idx


_E8P_PACKED_ABS_CACHED = get_packed_abs_grid()
_E8P_GRID, _E8P_GRID_IDX, _PARITY_IDX = get_full_grid(_E8P_PACKED_ABS_CACHED)


class QuipSharpQuestQuantizer(BaseQuantizer):
    def __init__(self, group_size=128, clip_factor=1.0, masking="none", debias=False):
        super().__init__(2.0)
        self.group_size = group_size
        self.clip_factor = clip_factor
        self.masking = masking
        self.debias = debias
        
        # QuIP# Logic from https://raw.githubusercontent.com/Cornell-RelaxML/quip-sharp/refs/heads/main/lib/codebook/latticee8_padded12.py
        self.codesz = _E8P_CODESZ
        self.register_buffer('grid_packed_abs', _E8P_PACKED_ABS_CACHED.cuda(), persistent=False)
        self.register_buffer('grid', _E8P_GRID.cuda(), persistent=False)
        self.register_buffer('grid_norm', _E8P_GRID.cuda().norm(dim=-1)**2, persistent=False)
        grid_part = _E8P_GRID[_PARITY_IDX].cuda() + 0.25
        grid_part = grid_part[
            torch.where(
                ((grid_part[:, :7] < 0).sum(dim=-1) <= 1) * \
                (grid_part[:, :7].min(dim=-1).values >= -0.5)
            )[0]]
        self.register_buffer('grid_part', grid_part, persistent=False)
        self.register_buffer('grid_part_norm', grid_part.norm(dim=-1)**2, persistent=False)
        abs_grid = get_abs_grid().cuda()
        self.register_buffer('grid_abs_odd', abs_grid.sum(dim=-1) % 2 == 1, persistent=False)
        self.register_buffer(
            'part_abs_map',
            self.round(grid_part.abs(), abs_grid,
                        abs_grid.norm(dim=-1)**2)[1],
            persistent=False)
        self.register_buffer('bit_map', 2**torch.arange(8, device="cuda"), persistent=False)
        self.grid_max_abs = self.grid_norm.max()**0.5

    def forward(self, x):
        orig_shape = x.shape
        
        x = x.reshape(-1, self.group_size)
        
        with torch.no_grad():
            std = torch.std(x, correction=0, dim=-1, keepdim=True)
            x_normalized = x / std
            x_q = self.quantize(
                x_normalized.view(-1, self.codesz), return_idx=False,
            ).view_as(x_normalized)
            
            if self.debias:
                num = (x_normalized * x_normalized).sum(dim=-1, keepdim=True)
                denom = (x_normalized * x_q).sum(dim=-1, keepdim=True)
                scales_fix = num / denom
                scales_fix = torch.where(scales_fix.isnan(), 1.0, scales_fix)
                x_q = x_q * scales_fix * std
            else:
                x_q = x_q * std
        
        if self.masking == "l1":
            # Square mask
            pass_mask = torch.logical_and(
                x_normalized >= self.grid_max_abs * self.clip_factor,
                x_normalized <= self.grid_max_abs * self.clip_factor,
            )
            x_pass = x * pass_mask.to(x.dtype)
        elif self.masking == "l2":
            # Circle mask
            pass_mask = (x_normalized.view(-1, self.codesz).norm(dim=-1, keepdim=True) <= self.grid_max_abs * self.clip_factor)
            x_pass = (
                x.view(-1, self.codesz) * pass_mask.to(x.dtype)
            ).view_as(x)
        elif self.masking == "none":
            x_pass = x
        
        return (x_pass + (x_q - x_pass).detach()).reshape(*orig_shape)
    
    def round(self, X, grid, grid_norm):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx

    def fast_quantize_part(self, X, parity):
        X_part = torch.abs(X)
        X_odd = torch.where((X < 0).sum(dim=-1) % 2 != 0)[0]
        X_part[X_odd, 7] = -X_part[X_odd, 7]
        mask = 1 - 2 * (X < 0).to(torch.float32)
        mask[X_odd, 7] = -mask[X_odd, 7]
        roundout, Xqidx = self.round(X_part, self.grid_part,
                                     self.grid_part_norm)
        vals = roundout * mask
        err = (X - vals).norm(dim=-1)
        abs_idx = self.part_abs_map[Xqidx]
        sign_mask = (((roundout < 0) ^ (mask < 0))[:,
                                                   [0, 2, 4, 6, 1, 3, 5, 7]])
        sign_mask[:, 7] = sign_mask[:, 7] ^ self.grid_abs_odd[abs_idx]
        sign_mask[:, 0] = sign_mask[:, 0] ^ parity
        mask_idx = (sign_mask * self.bit_map).sum(dim=-1).int()
        idx = (abs_idx << 8) + mask_idx
        return vals, idx, err

    def quantize(self, X, return_idx=True, **kwargs):
        X_plus = X + 1 / 4  # quantize X to D8^ - 1/4
        X_minus = X - 1 / 4  # quantize X to D8^ + 1/4

        plus_vals, plus_idx, plus_err = self.fast_quantize_part(X_plus, True)
        minus_vals, minus_idx, minus_err = self.fast_quantize_part(
            X_minus, False)

        which = plus_err < minus_err
        final_vals = torch.where(which.unsqueeze(-1), plus_vals - 1 / 4,
                                 minus_vals + 1 / 4)
        final_idx = torch.where(which, plus_idx, minus_idx)

        if return_idx:
            return final_vals, final_idx

        return final_vals
