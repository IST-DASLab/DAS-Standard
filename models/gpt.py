import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from argparse import Namespace as ArgParseNamespace


class KarpathyLayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ArgParseNamespace, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id

        E = cfg.model.embed_size
        bias = cfg.model.bias

        # TODO: introduce QuartetLinear
        self.Wq = nn.Linear(E, E, bias=bias)
        self.Wk = nn.Linear(E, E, bias=bias)
        self.Wv = nn.Linear(E, E, bias=bias)
        self.Wo = nn.Linear(E, E, bias=bias)
        self.drop_res = nn.Dropout(cfg.model.dropout)

        self.is_flash_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.is_flash_attn:
            print(f'WARNING: SDPA attention not found, using manual implementation')
            T = cfg.model.seq_len

            mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
            self.register_buffer('mask', mask)

            self.drop_attn = nn.Dropout(cfg.model.dropout)

    def forward(self, x):
        B, T, E = x.shape
        H = self.cfg.model.num_heads

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        #                                             Q/K/V        after view        after transpose
        Q = Q.view(B, T, H, E // H).transpose(1, 2) # (B, T, E) => (B, T, H, E/H) => (B, H, T, E/H)
        K = K.view(B, T, H, E // H).transpose(1, 2) # (B, T, E) => (B, T, H, E/H) => (B, H, T, E/H)
        V = V.view(B, T, H, E // H).transpose(1, 2) # (B, T, E) => (B, T, H, E/H) => (B, H, T, E/H)

        if self.is_flash_attn:
            y = torch.nn.functional.scaled_dot_product_attention(
                query=Q,
                key=K,
                value=V,
                attn_mask=None,
                dropout_p=self.cfg.model.dropout,
                is_causal=True,
            )
        else: # manual implementation of attention
            attn = (Q @ K.transpose(-2, -1)) / math.sqrt(K.size(-1))
            attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf")) # write -inf where mask is zero
            attn = F.softmax(attn, dim=-1)
            attn = self.drop_attn(attn)
            y = attn @ V

        y = y.transpose(1, 2).contiguous().view(B, T, E)
        y = self.Wo(y)
        y = self.drop_res(y)
        return y


class MLP(nn.Module):
    def __init__(self, cfg: ArgParseNamespace, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id

        E = cfg.model.embed_size
        self.fc = nn.Linear(in_features=E, out_features=4 * E)
        self.Wo = nn.Linear(in_features=4 * E, out_features=E)
        self.dropout = nn.Dropout(cfg.model.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = self.Wo(x)
        x = self.dropout(x)
        return x


class BlockGPT(nn.Module):
    def __init__(self, cfg: ArgParseNamespace, layer_id: int):
        super().__init__()
        self.norm1 = nn.LayerNorn(cfg.model.embed_size)
        self.attn = CausalSelfAttention(cfg, layer_id)
        self.norm2 = KarpathyLayerNorm(cfg.model.embed_size)
        self.mlp = MLP(cfg, layer_id)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ArgParseNamespace):
        super().__init__()
        self.cfg = cfg

        self.tok_embed = nn.Embedding(cfg.model.vocab_size, cfg.model.embed_size)
        self.pos_embed = nn.Embedding(cfg.model.seq_len, cfg.model.embed_size)
        self.dropout = nn.Dropout(cfg.model.dropout)
        self.blocks = nn.ModuleList([
            BlockGPT(cfg, layer_id)
            for layer_id in range(cfg.model.num_layers)
        ])
        self.norm = KarpathyLayerNorm(cfg.model.embed_size, bias=cfg.model.bias)
        self.lm_head = nn.Linear(cfg.model.embed_size, cfg.model.vocab_size, bias=cfg.model.bias)

        if cfg.model.weight_tying:
            self.tok_embed.weight = self.lm_head.weight

        # initialize weights
        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if name.endswith("Wo.weight"):
                torch.nn.init.normal_(p, mean=0, std=0.02 / math.sqrt(2 * cfg.model.num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, inputs):
        B, T = inputs.shape
        device = inputs.device

        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.tok_embed(inputs)
        pos_emb = self.pos_embed(pos)

        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.lm_head(x)
        return x

        # if targets is None:
        #     # inference-time mini-optimization: forward only through lm_head on the very last position
        #     logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        #     loss = None
        # else:
        #     logits = self.lm_head(x)
        #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        #
        # return logits, loss

    def get_num_params(self):
        embed_params_count = self.tok_embed.weight.numel()
        non_embed_params_count = sum(p.numel() for p in self.blocks.parameters())
        if not self.cfg.model.weight_tying:
            embed_params_count *= 2
        return embed_params_count, non_embed_params_count
