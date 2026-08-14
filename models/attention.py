import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from argparse import Namespace as ArgParseNamespace

from models.rope import precompute_rope, apply_rope


class MultiHeadSelfAttention(nn.Module):
    @staticmethod
    def get_param_count(E):
        return 4 * E * E

    def __init__(self, cfg: ArgParseNamespace, layer_id: int, use_rope: bool=False):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.use_rope = use_rope

        E = cfg.model.embed_size
        S = cfg.model.seq_len
        bias = cfg.model.bias
        drp = cfg.model.dropout

        self.Wq = nn.Linear(E, E, bias=bias)
        self.Wk = nn.Linear(E, E, bias=bias)
        self.Wv = nn.Linear(E, E, bias=bias)
        self.Wo = nn.Linear(E, E, bias=bias)
        self.dropout_attn = nn.Dropout(p=drp)
        self.dropout_out = nn.Dropout(p=drp)
        self.register_buffer(
            name="mask",
            tensor=torch.triu(torch.ones(S, S), diagonal=1).bool().view(1, 1, S, S)
        )

        if use_rope:
            cos, sin = precompute_rope(cfg)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)

    def forward(self, x):
        B, T, E = x.shape
        Nh = self.cfg.model.num_heads
        H = E // Nh # head size

        Q = self.Wq(x) # (B, T, E)
        K = self.Wk(x) # (B, T, E)
        V = self.Wv(x) # (B, T, E)

        # (B, T, E) => (B, T, Nh, H) => (B, Nh, T, H)
        Q = Q.view(B, T, Nh, H).transpose(1, 2)
        K = K.view(B, T, Nh, H).transpose(1, 2)
        V = V.view(B, T, Nh, H).transpose(1, 2)

        if self.use_rope:
            Q = apply_rope(Q, self.cos, self.sin)
            K = apply_rope(K, self.cos, self.sin)

        # (B, Nh, T, H) @ (B, Nh, T, H).T(2, 3) = (B, Nh, T, T)
        A = Q @ K.transpose(2, 3) / math.sqrt(K.size(-1))
        A = A.masked_fill(self.mask[:, :, :T, :T], float('-inf'))
        A = F.softmax(A, dim=-1)
        A = self.dropout_attn(A)

        # (B, Nh, T, T) @ (B, Nh, T, H) = (B, Nh, T, H)
        out = A @ V

        # (B, Nh, T, H).T(1,2) = (B, T, Nh, H) => (B, T, E)
        out = out.transpose(1, 2).contiguous().view(B, T, E)
        out = self.Wo(out)
        out = self.dropout_out(out)
        return out

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ArgParseNamespace, layer_id: int, use_rope: bool=False):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.use_rope = use_rope

        S = cfg.model.seq_len
        E = cfg.model.embed_size
        Nh = cfg.model.num_heads
        H = E // Nh # head size
        bias = cfg.model.bias
        Nkvg = cfg.model.attn_kv_groups
        drp = cfg.model.dropout

        self.Wq = nn.Linear(E, E, bias=bias)
        self.Wk = nn.Linear(E, Nkvg * H, bias=bias)
        self.Wv = nn.Linear(E, Nkvg * H, bias=bias)
        self.Wo = nn.Linear(E, E, bias=bias)
        self.dropout_attn = nn.Dropout(p=drp)
        self.dropout_out = nn.Dropout(p=drp)

        self.register_buffer(
            name="mask",
            tensor=torch.triu(torch.ones(S, S), diagonal=1).bool().view(1, 1, S, S)
        )

        if use_rope:
            cos, sin = precompute_rope(cfg)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)

    def forward(self, x):
        B, T, E = x.shape
        Nh = self.cfg.model.num_heads
        H = E // Nh # head size
        Nkvg = self.cfg.model.attn_kv_groups
        Gs = Nh // Nkvg

        Q = self.Wq(x) # (B, T, E = Nh * H)
        K = self.Wk(x) # (B, T, Nkvg * H)
        V = self.Wv(x) # (B, T, Nkvg * H)

        # (B, T, E=Nh*H) => (B, T, Nh, H).T(1,2) = (B, Nh, T, H)
        Q = Q.view(B, T, Nh, H).transpose(1, 2)

        # (B, T, Nkvg*H) => (B, T, Nkvg, H).T(1,2) = (B, Nkvg, T, H)
        K = K.view(B, T, Nkvg, H).transpose(1, 2)
        V = V.view(B, T, Nkvg, H).transpose(1, 2)

        # (B, Nkvg, T, H) => (B, Nkvg*Gs, T, H): [x1, x2] becomes [x1, x1, x2, x2]
        K = K.repeat_interleave(Gs, dim=1)
        V = V.repeat_interleave(Gs, dim=1)

        if self.use_rope:
            Q = apply_rope(Q, self.cos, self.sin)
            K = apply_rope(K, self.cos, self.sin)

        # (B, Nh, T, H) @ (B, Nkvg*Gs=Nh, T, H).T(2,3) = (B, Nh, T, T)
        A = Q @ K.transpose(2, 3) / math.sqrt(K.shape[-1])
        A = A.masked_fill(self.mask[:, :, :T, :T], float('-inf'))
        A = F.softmax(A, dim=-1)
        A = self.dropout_attn(A)

        # (B, Nh, T, T) @ (B, Nh=Nkvg*Gs, T, H) = (B, Nh, T, H)
        out = A @ V

        # (B, Nh, T, H).T(1,2) = (B, T, Nh, H) = (B, T, E)
        out = out.transpose(1, 2).contiguous().view(B, T, E)
        out = self.Wo(out)
        out = self.dropout_out(out)
        return out
