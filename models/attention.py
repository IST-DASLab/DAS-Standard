import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from argparse import Namespace as ArgParseNamespace

from models.rope import apply_rope

def attn_factory(cfg: ArgParseNamespace):
    match cfg.model.attn_type:
        case "mhsa":
            return MultiHeadSelfAttention(cfg)
        # case "mhsa-flash":
        #     raise NotImplementedError
        case "gqa":
            return GroupedQueryAttention(cfg)
        # case "gqa-flash":
        #     raise NotImplementedError
        case _:
            raise NotImplementedError


class MultiHeadSelfAttention(nn.Module):
    @staticmethod
    def get_param_count(cfg: ArgParseNamespace):
        E = cfg.model.embed_size
        return 4 * E * E

    def __init__(self, cfg: ArgParseNamespace):
        super().__init__()
        self.cfg = cfg

        E = cfg.model.embed_size
        bias = cfg.model.bias
        drp = cfg.model.dropout

        self.Wq = nn.Linear(E, E, bias=bias)
        self.Wk = nn.Linear(E, E, bias=bias)
        self.Wv = nn.Linear(E, E, bias=bias)
        self.Wo = nn.Linear(E, E, bias=bias)
        self.dropout_attn = nn.Dropout(p=drp)
        self.dropout_out = nn.Dropout(p=drp)

    def forward(self, x, mask, cos, sin):
        assert mask is not None, "Found mask=None in MHSA!"
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

        if (cos is not None) and (sin is not None):
            Q = apply_rope(Q, cos, sin)
            K = apply_rope(K, cos, sin)

        # (B, Nh, T, H) @ (B, Nh, T, H).T(2, 3) = (B, Nh, T, T)
        A = Q @ K.transpose(2, 3) / math.sqrt(K.size(-1))
        A = A.masked_fill(mask[:, :, :T, :T], float('-inf'))
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
    @staticmethod
    def get_param_count(cfg: ArgParseNamespace):
        E = cfg.model.embed_size
        Nh = cfg.model.num_heads
        H = E // Nh # head size
        Nkvg = cfg.model.attn_kv_groups
        return 2 * (E * E + E * (Nkvg * H))

    def __init__(self, cfg: ArgParseNamespace):
        super().__init__()
        self.cfg = cfg

        E = cfg.model.embed_size
        Nh = cfg.model.num_heads
        H = E // Nh # head size
        bias = cfg.model.bias
        Nkvg = cfg.model.attn_kv_groups
        drp = cfg.model.dropout

        assert E % Nh == 0, "Embedding size must be a multiple of Nh"
        assert Nh % Nkvg == 0, "Number of heads must be a multiple of number KV-groups"

        self.Wq = nn.Linear(E, E, bias=bias)
        self.Wk = nn.Linear(E, Nkvg * H, bias=bias)
        self.Wv = nn.Linear(E, Nkvg * H, bias=bias)
        self.Wo = nn.Linear(E, E, bias=bias)
        self.dropout_attn = nn.Dropout(p=drp)
        self.dropout_out = nn.Dropout(p=drp)

    def forward(self, x, mask, cos, sin):
        assert mask is not None, "Found mask=None in GQA!"
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

        if (cos is not None) and (sin is not None):
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
