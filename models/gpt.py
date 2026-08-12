import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from argparse import Namespace as ArgParseNamespace

from models.attention import MultiHeadSelfAttention
from models.norm import KarpathyLayerNorm


class GPTFeedForward(nn.Module):
    @staticmethod
    def get_param_count(E, exp_factor):
        return 2 * E * (E * exp_factor)

    def __init__(self, cfg: ArgParseNamespace, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id

        E = cfg.model.embed_size
        E_expanded = E * cfg.model.mlp_expansion_factor
        bias = cfg.model.bias

        self.fc = nn.Linear(in_features=E, out_features=E_expanded, bias=bias)
        self.Wo = nn.Linear(in_features=E_expanded, out_features=E, bias=bias)
        self.dropout = nn.Dropout(cfg.model.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = self.Wo(x)
        x = self.dropout(x)
        return x


class GPTBlock(nn.Module):
    def __init__(self, cfg: ArgParseNamespace, layer_id: int):
        super().__init__()
        self.norm1 = KarpathyLayerNorm(cfg.model.embed_size, bias=cfg.model.bias)
        self.attn = MultiHeadSelfAttention(cfg, layer_id)
        self.norm2 = KarpathyLayerNorm(cfg.model.embed_size, bias=cfg.model.bias)
        self.mlp = GPTFeedForward(cfg, layer_id)

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
            GPTBlock(cfg, layer_id)
            for layer_id in range(cfg.model.num_layers)
        ])
        self.norm = KarpathyLayerNorm(cfg.model.embed_size, bias=cfg.model.bias)
        self.lm_head = nn.Linear(cfg.model.embed_size, cfg.model.vocab_size, bias=cfg.model.bias)

        if cfg.model.tie_embeds:
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

    def forward(self, inputs, targets):
        B, T = inputs.shape
        device = inputs.device

        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.tok_embed(inputs)
        pos_emb = self.pos_embed(pos)

        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = F.cross_entropy(
            input=logits.view(-1, logits.size(-1)),
            target=targets.view(-1),
            ignore_index=-1 # not sure about this
        )
        return loss

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
        if not self.cfg.model.tie_embeds:
            embed_params_count *= 2
        return embed_params_count, non_embed_params_count
