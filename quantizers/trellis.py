import itertools
import math
from functools import cache

import torch
from torch import nn

from .base import BaseQuantizer


def decode_3inst(x):

    def bfe16_to_fp16(x):
        x[torch.where(x >= 2**15)] -= 2**16
        return x.to(torch.int16).view(torch.float16)

    a = 89226354
    b = 64248484
    fpmask = 996162400
    x = x.to(torch.int64)
    x = x & ((1 << 32) - 1)
    x = x * a + b
    mask = (1 << 15) + ((1 << 12) - 1)
    mask = (mask << 16) + mask
    res = (mask & x) ^ fpmask
    top = bfe16_to_fp16(res >> 16)
    bottom = bfe16_to_fp16(res & ((1 << 16) - 1))
    return (top + bottom).float()


def quantlut(tlut, L, nbits):
    with torch.no_grad():
        lut = torch.arange(1 << L)
        lut = (lut + 1) * lut
        lut = (lut >> (16 - nbits)) & ((1 << nbits) - 1)
    lut = tlut[lut]
    return lut


def quantlut_sym(tlut, L, nbits):
    with torch.no_grad():
        lut = torch.arange(1 << L, device=tlut.device)
        lut = (lut + 1) * lut
        sflp = 1 - ((lut >> 15) & 1) * 2
        lut = (lut >> (16 - nbits - 1)) & ((1 << nbits) - 1)
    lut = tlut[lut]
    lut[:, 0] = lut[:, 0] * sflp
    return lut


class QtipQuestQuantizer(BaseQuantizer):
    def __init__(self, l=16, k=2, v=1, tile_shape=(16, 16), group_size=128, debias=False):
        super().__init__(k / v)
        
        self.l = l
        self.k = k
        self.v = v
        self.tile_shape=tile_shape
        
        self.group_size = group_size
        self.debias = debias
        
        assert self.v == 1
        self.register_buffer(
            'lut',
            decode_3inst(torch.arange(2**l, device="cuda")).unsqueeze(0),
            persistent=False,
        )
    
        self.fakeinf = torch.tensor(torch.inf, device="cuda")

        self.register_buffer(
            'sumdelta',
            torch.arange(2**(k * v), device="cuda") << (l - k * v),
            persistent=False,
        )
        self.sumdelta = self.sumdelta.view(1, 1, -1)

        self.register_buffer('state', torch.arange(2**l, device="cuda").unsqueeze(0), persistent=False)
        self.register_buffer(
            'state_cand',
            (self.state >> (k * v))[0, ::2**(k * v)].unsqueeze(-1) + self.sumdelta,
            persistent=False,
        )
        self.register_buffer('recons_state', self.recons(self.state), persistent=False)

        self.input_version = -1
        self.cached_output = None

    def recons(self, encoded):
        return self.lut[
            :,
            encoded.int().to(self.lut.device),
        ].to(encoded.device)

    def update(self, cost, thing):
        state_err = (self.recons_state -
                     thing.unsqueeze(-1)).square().sum(dim=0)
        cand_cost = torch.gather(
            cost.unsqueeze(-2).expand(-1, self.state_cand.shape[1], -1), -1,
            self.state_cand.expand(len(cost), -1, 2**(self.k * self.v)))
        best = torch.min(cand_cost, dim=-1)
        cost = state_err + best.values.unsqueeze(-1).expand(
            -1, -1, 2**(self.k * self.v)).reshape(state_err.shape)
        prev_state = torch.gather(
            self.state_cand.expand(thing.shape[1], -1, -1), -1,
            best.indices.unsqueeze(-1))[..., 0]
        return prev_state, cost

    def viterbi(self, X, overlap=None):
        T, B = X.shape
        assert T % self.v == 0
        # cost is (B, 2**L)
        cost = (self.recons_state -
                X[:self.v].unsqueeze(-1)).square().sum(dim=0)

        if overlap is not None:
            mask = torch.ones(B, 2**self.l, device=X.device) * self.fakeinf
            allow = (overlap <<
                     (self.k * self.v)).unsqueeze(-1) + torch.arange(
                         2**(self.k * self.v)).to(X.device).view(1, 1, -1)
            mask.scatter_(1, allow[0], 0)
            cost = torch.min(cost + mask, self.fakeinf)

        from_state = torch.zeros(T // self.v,
                                 B,
                                 2**(self.l - self.k * self.v),
                                 dtype=self.state.dtype,
                                 device=self.state.device)

        for i in range(1, T // self.v):
            from_state[i], cost = self.update(cost,
                                              X[i * self.v:(i + 1) * self.v])

        if overlap is not None:
            mask = torch.ones(B, 2**self.l, device=X.device) * self.fakeinf
            allow = (overlap.unsqueeze(-1) + self.sumdelta.unsqueeze(0))
            mask.scatter_(1, allow[0, 0], 0)
            cost = torch.min(cost + mask, self.fakeinf)

        final_state = torch.zeros(T // self.v,
                                  B,
                                  dtype=torch.int32,
                                  device=X.device)
        final_state[T // self.v - 1] = torch.argmin(cost, dim=-1)
        for i in range(T // self.v - 1, 0, -1):
            final_state[i - 1] = torch.gather(
                from_state[i], -1,
                (final_state[i].to(torch.int64).unsqueeze(-1)) >>
                (self.k * self.v))[..., 0]
        return final_state

    def quantize_seq(self, X, overlap=None, **kwargs):
        T, NO = X.shape
        bs = min(2**(24 - self.l), NO)
        pad_amt = math.ceil(NO / bs) * bs - NO
        X = torch.nn.functional.pad(X, (0, pad_amt))
        T, N = X.shape
        X = X.reshape(T, N // bs, bs).transpose(0, 1).contiguous()
        if overlap is not None:
            overlap = torch.nn.functional.pad(overlap, (0, pad_amt))
            overlap = overlap.reshape(N // bs, bs)

        Qidxs = torch.zeros(N // bs,
                            T // self.v,
                            bs,
                            dtype=torch.int32,
                            device=X.device)
        for i in range(len(X)):
            b_overlap = None if overlap is None else overlap[i]
            Qidxs[i] = self.viterbi(X[i], overlap=b_overlap)
        Qidxs = Qidxs.transpose(0, 1).reshape(T // self.v, N)[:, :NO]
        return Qidxs

    def quantize_block(self, X):
        X = X.T.contiguous().to(torch.float16)
        T = X.shape[0]
        roll_X = torch.roll(X, T // (2 * self.v) * self.v, 0)
        state = self.quantize_seq(roll_X, overlap=None)
        overlap = state[T // (2 * self.v)] >> self.k * self.v
        state = self.quantize_seq(X, overlap=overlap)
        hatX = self.recons(state).transpose(0, 1).reshape(X.shape)
        return hatX.T.contiguous().to(X.device), state.T.contiguous().to(
            X.device)
    
    @torch.compiler.disable()
    def quantize(self, x):
        orig_shape = x.shape
        
        x = x.reshape(
            orig_shape[0] // self.tile_shape[0], self.tile_shape[0],
            orig_shape[1] // self.tile_shape[1], self.tile_shape[1],
        ).permute(0, 2, 1, 3).reshape(-1, self.tile_shape[0] * self.tile_shape[1]).contiguous()
        
        xq, _ = self.quantize_block(x)
        
        xq = xq.reshape(
            orig_shape[0] // self.tile_shape[0], orig_shape[1] // self.tile_shape[1],
            self.tile_shape[0], self.tile_shape[1],
        ).permute(0, 2, 1, 3).reshape(*orig_shape).contiguous()
        
        return xq

    def forward(self, x):
        assert not x.requires_grad
        
        if self.cached_output is not None:
            return self.cached_output
        self.input_version = x._version
        
        orig_shape = x.shape
        
        x = x.reshape(-1, self.group_size)
        
        with torch.no_grad():
            std = torch.std(x, correction=0, dim=-1, keepdim=True)
            x_normalized = x / std
            x_q = self.quantize(
                x_normalized.view(*orig_shape),
            ).view_as(x_normalized)
            x_q = x_q.view_as(x_normalized)
            
            if self.debias:
                num = (x_normalized * x_normalized).sum(dim=-1, keepdim=True)
                denom = (x_normalized * x_q).sum(dim=-1, keepdim=True)
                scales_fix = num / denom
                scales_fix = torch.where(scales_fix.isnan(), 1.0, scales_fix)
                x_q = x_q * scales_fix * std
            else:
                x_q = x_q * std
            
        self.cached_output = x_q.clone().detach().reshape(*orig_shape)
        
        return self.cached_output
