import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from argparse import Namespace as ArgParseNamespace

from models.norm import RMSNorm
from models.rope import apply_rope

# def attn_factory(cfg: ArgParseNamespace):
#     attn_type = cfg.model.attn_type
#     if "mhsa" in attn_type:
#         return MultiHeadSelfAttention(cfg)
#     elif "gqa" in attn_type:
#         return GroupedQueryAttention(cfg)
#     else:
#         raise NotImplementedError(f"Unknown attention type: {attn_type}")

def get_GQA_param_count(cfg: ArgParseNamespace):
    E = cfg.model.embed_size
    Nh = cfg.model.num_heads
    H = E // Nh # head size
    Nkvg = cfg.model.attn_gqa_kvg

    qk_norm = cfg.model.qk_norm

    params_linear = 2 * (E * E + E * (Nkvg * H))
    params_norm = 2 * E if qk_norm else 0 # 2E for Q and K norms (only weight, no bias)

    return params_linear + params_norm

def get_MHSA_param_count(cfg: ArgParseNamespace):
    E = cfg.model.embed_size
    qk_norm = cfg.model.qk_norm
    params_linear = 4 * E * E
    params_norm = 2 * E if qk_norm else 0 # 2E for Q and K norms (only weight, no bias)
    return params_linear + params_norm

class Attention(nn.Module):
    def __init__(self, cfg: ArgParseNamespace):
        super().__init__()
        self.cfg = cfg

        E = cfg.model.embed_size
        Nh = cfg.model.num_heads
        Nkvg = cfg.model.attn_gqa_kvg
        H = E // Nh # head size
        bias = cfg.model.bias
        # drp = cfg.model.dropout
        qk_norm = cfg.model.qk_norm

        self.is_mhsa = 'mhsa' in cfg.model.attn_type
        self.is_gqa = 'gqa' in cfg.model.attn_type
        self.is_flash = 'flash' in cfg.model.attn_type

        if self.is_mhsa:
            out_kv = E
            self.KV_view_dim = Nh
        elif self.is_gqa:
            out_kv = Nkvg * E
            self.KV_view_dim = Nkvg
        else:
            raise NotImplementedError(f"Unknown attention type: {cfg.model.attn_type}")

        self.Wq = nn.Linear(in_features=E, out_features=E, bias=bias)
        self.Wk = nn.Linear(in_features=E, out_features=out_kv, bias=bias)
        self.Wv = nn.Linear(in_features=E, out_features=out_kv, bias=bias)
        self.Wo = nn.Linear(in_features=E, out_features=E, bias=bias)
        # self.dropout_attn = nn.Dropout(p=drp)
        # self.dropout_out = nn.Dropout(p=drp)

        if qk_norm:
            self.q_norm = RMSNorm(ndim=H, bias=False)
            self.k_norm = RMSNorm(ndim=H, bias=False)

    def _compute_pre_attn(self, x, cos, sin):
        B, T, E = x.shape
        Nh = self.cfg.model.num_heads
        qk_norm = self.cfg.model.qk_norm
        H = E // Nh # head size
        Nkvg = self.cfg.model.attn_gqa_kvg
        Gs = Nh // Nkvg

        Q = self.Wq(x)  # (B, T, E)
        K = self.Wk(x)  # (B, T, E)
        V = self.Wv(x)  # (B, T, E)

        # (B, T, E) => (B, T, Nh, H) => (B, Nh, T, H)
        Q = Q.view(B, T, Nh, H).transpose(1, 2)
        K = K.view(B, T, self.KV_view_dim, H).transpose(1, 2)
        V = V.view(B, T, self.KV_view_dim, H).transpose(1, 2)

        #####
        ##### QK-norm
        #####
        if qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        if (cos is not None) and (sin is not None):
            Q = apply_rope(Q, cos, sin)
            K = apply_rope(K, cos, sin)

        if self.is_gqa and not self.is_flash:
            # if flash, then set the flag in SDPA implementation, which will apply the interleave internally for GQA
            # (B, Nkvg, T, H) => (B, Nkvg*Gs, T, H): [x1, x2] becomes [x1, x1, x2, x2]
            K = K.repeat_interleave(Gs, dim=1)
            V = V.repeat_interleave(Gs, dim=1)

        return Q, K, V

    def _compute_attn(self, mask, Q, K, V):
        if self.is_flash:
            attn_out = F.scaled_dot_product_attention(
                query=Q,
                key=K,
                value=V,
                attn_mask=None,
                is_causal=True,
                scale=None, # if None, then uses 1/sqrt(E)
                enable_gqa=self.is_gqa,
            )
        else:
            T = mask.size(-1)
            # (B, Nh, T, H) @ (B, Nh, T, H).T(2, 3) = (B, Nh, T, T)
            A = Q @ K.transpose(2, 3) / math.sqrt(K.size(-1))
            A = A.masked_fill(mask[:, :, :T, :T], float('-inf'))
            A = F.softmax(A, dim=-1)
            # A = self.dropout_attn(A) # removed

            # (B, Nh, T, T) @ (B, Nh, T, H) = (B, Nh, T, H)
            attn_out = A @ V

        return attn_out

    def _compute_post_attn(self, x, attn_out):
        # B, T, E = x.shape # x is given as parameter just to extract shapes

        # (B, Nh, T, H).T(1,2) = (B, T, Nh, H) => (B, T, E)
        out = attn_out.transpose(1, 2).contiguous().view_as(x) # .view(B, T, E)
        out = self.Wo(out)
        # out = self.dropout_out(out)

        return out

    def forward(self, x, mask, cos, sin):
        assert mask is not None, "Found mask=None in MHSA!"
        Q, K, V = self._compute_pre_attn(x, cos, sin)
        out = self._compute_attn(mask, Q, K, V)
        out = self._compute_post_attn(x, out)
        return out

class MultiHeadSelfAttention(nn.Module):
    @staticmethod
    def get_param_count(cfg: ArgParseNamespace):
        E = cfg.model.embed_size
        qk_norm = cfg.model.qk_norm
        params_linear = 4 * E * E
        params_norm = 2 * E if qk_norm else 0 # 2E for Q and K norms (only weight, no bias)
        return params_linear + params_norm

    def __init__(self, cfg: ArgParseNamespace):
        super().__init__()
        self.cfg = cfg

        E = cfg.model.embed_size
        Nh = cfg.model.num_heads
        H = E // Nh # head size
        bias = cfg.model.bias
        # drp = cfg.model.dropout
        qk_norm = cfg.model.qk_norm

        self.Wq = nn.Linear(in_features=E, out_features=E, bias=bias)
        self.Wk = nn.Linear(in_features=E, out_features=E, bias=bias)
        self.Wv = nn.Linear(in_features=E, out_features=E, bias=bias)
        self.Wo = nn.Linear(in_features=E, out_features=E, bias=bias)
        # self.dropout_attn = nn.Dropout(p=drp)
        # self.dropout_out = nn.Dropout(p=drp)

        if qk_norm:
            self.q_norm = RMSNorm(ndim=H, bias=False)
            self.k_norm = RMSNorm(ndim=H, bias=False)

    def forward(self, x, mask, cos, sin):
        assert mask is not None, "Found mask=None in MHSA!"
        B, T, E = x.shape
        Nh = self.cfg.model.num_heads
        qk_norm = self.cfg.model.qk_norm
        H = E // Nh # head size

        Q = self.Wq(x) # (B, T, E)
        K = self.Wk(x) # (B, T, E)
        V = self.Wv(x) # (B, T, E)

        # (B, T, E) => (B, T, Nh, H) => (B, Nh, T, H)
        Q = Q.view(B, T, Nh, H).transpose(1, 2)
        K = K.view(B, T, Nh, H).transpose(1, 2)
        V = V.view(B, T, Nh, H).transpose(1, 2)

        #####
        ##### QK-norm
        #####
        if qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        if (cos is not None) and (sin is not None):
            Q = apply_rope(Q, cos, sin)
            K = apply_rope(K, cos, sin)

        # (B, Nh, T, H) @ (B, Nh, T, H).T(2, 3) = (B, Nh, T, T)
        A = Q @ K.transpose(2, 3) / math.sqrt(K.size(-1))
        A = A.masked_fill(mask[:, :, :T, :T], float('-inf'))
        A = F.softmax(A, dim=-1)
        # A = self.dropout_attn(A)

        # (B, Nh, T, T) @ (B, Nh, T, H) = (B, Nh, T, H)
        out = A @ V

        # (B, Nh, T, H).T(1,2) = (B, T, Nh, H) => (B, T, E)
        out = out.transpose(1, 2).contiguous().view(B, T, E)
        out = self.Wo(out)
        # out = self.dropout_out(out)
        return out


class GroupedQueryAttention(nn.Module):
    @staticmethod
    def get_param_count(cfg: ArgParseNamespace):
        E = cfg.model.embed_size
        Nh = cfg.model.num_heads
        H = E // Nh # head size
        Nkvg = cfg.model.attn_gqa_kvg

        qk_norm = cfg.model.qk_norm

        params_linear = 2 * (E * E + E * (Nkvg * H))
        params_norm = 2 * E if qk_norm else 0 # 2E for Q and K norms (only weight, no bias)

        return params_linear + params_norm

    def __init__(self, cfg: ArgParseNamespace):
        super().__init__()
        self.cfg = cfg

        E = cfg.model.embed_size
        Nh = cfg.model.num_heads
        H = E // Nh # head size
        bias = cfg.model.bias
        Nkvg = cfg.model.attn_gqa_kvg
        # drp = cfg.model.dropout
        qk_norm = cfg.model.qk_norm

        assert E % Nh == 0, "Embedding size must be a multiple of Nh"
        assert Nh % Nkvg == 0, "Number of heads must be a multiple of number KV-groups"

        self.Wq = nn.Linear(in_features=E, out_features=E, bias=bias)
        self.Wk = nn.Linear(in_features=E, out_features=Nkvg * H, bias=bias)
        self.Wv = nn.Linear(in_features=E, out_features=Nkvg * H, bias=bias)
        self.Wo = nn.Linear(in_features=E, out_features=E, bias=bias)
        # self.dropout_attn = nn.Dropout(p=drp)
        # self.dropout_out = nn.Dropout(p=drp)

        if qk_norm:
            self.q_norm = RMSNorm(ndim=H, bias=False)
            self.k_norm = RMSNorm(ndim=H, bias=False)

    def forward(self, x, mask, cos, sin):
        assert mask is not None, "Found mask=None in GQA!"
        B, T, E = x.shape
        Nh = self.cfg.model.num_heads
        H = E // Nh # head size
        Nkvg = self.cfg.model.attn_gqa_kvg
        Gs = Nh // Nkvg
        qk_norm = self.cfg.model.qk_norm

        Q = self.Wq(x) # (B, T, E = Nh * H)
        K = self.Wk(x) # (B, T, Nkvg * H)
        V = self.Wv(x) # (B, T, Nkvg * H)

        # (B, T, E=Nh*H) => (B, T, Nh, H).T(1,2) = (B, Nh, T, H)
        Q = Q.view(B, T, Nh, H).transpose(1, 2)

        # (B, T, Nkvg*H) => (B, T, Nkvg, H).T(1,2) = (B, Nkvg, T, H)
        K = K.view(B, T, Nkvg, H).transpose(1, 2)
        V = V.view(B, T, Nkvg, H).transpose(1, 2)

        #####
        ##### QK-norm
        #####
        if qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        #####
        ##### RoPE
        #####
        if (cos is not None) and (sin is not None):
            Q = apply_rope(Q, cos, sin)
            K = apply_rope(K, cos, sin)

        # (B, Nkvg, T, H) => (B, Nkvg*Gs, T, H): [x1, x2] becomes [x1, x1, x2, x2]
        K = K.repeat_interleave(Gs, dim=1)
        V = V.repeat_interleave(Gs, dim=1)

        # (B, Nh, T, H) @ (B, Nkvg*Gs=Nh, T, H).T(2,3) = (B, Nh, T, T)
        A = Q @ K.transpose(2, 3) / math.sqrt(K.size(-1))
        A = A.masked_fill(mask[:, :, :T, :T], float('-inf'))
        A = F.softmax(A, dim=-1)
        # A = self.dropout_attn(A)

        # (B, Nh, T, T) @ (B, Nh=Nkvg*Gs, T, H) = (B, Nh, T, H)
        out = A @ V

        # (B, Nh, T, H).T(1,2) = (B, T, Nh, H) = (B, T, E)
        out = out.transpose(1, 2).contiguous().view(B, T, E)
        out = self.Wo(out)
        # out = self.dropout_out(out)
        return out
