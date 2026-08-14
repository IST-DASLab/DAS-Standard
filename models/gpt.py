import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from argparse import Namespace as ArgParseNamespace

from models.attention import MultiHeadSelfAttention
from models.feedforward import GPTFeedForward
from models.norm import KarpathyLayerNorm
from models.rope import precompute_rope


class GPTBlock(nn.Module):
    def __init__(self, cfg: ArgParseNamespace):
        super().__init__()
        self.attn = MultiHeadSelfAttention(cfg)
        self.mlp = GPTFeedForward(cfg)
        self.norm1 = KarpathyLayerNorm(cfg.model.embed_size, bias=cfg.model.bias)
        self.norm2 = KarpathyLayerNorm(cfg.model.embed_size, bias=cfg.model.bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ArgParseNamespace):
        super().__init__()
        self.cfg = cfg

        E = cfg.model.embed_size
        V = cfg.model.vocab_size
        L = cfg.model.num_layers
        S = cfg.model.seq_len
        drp = cfg.model.dropout
        bias = cfg.model.bias

        self.tok_embed = nn.Embedding(V, E)
        self.pos_embed = nn.Embedding(S, E)
        self.dropout = nn.Dropout(drp)
        self.blocks = nn.ModuleList([GPTBlock(cfg) for _ in range(L)])
        self.norm = KarpathyLayerNorm(E, bias=bias)
        self.lm_head = nn.Linear(E, V, bias=bias)

        if cfg.model.tie_embeds:
            self.tok_embed.weight = self.lm_head.weight

        # initialize weights
        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if name.endswith("Wo.weight"):
                torch.nn.init.normal_(p, mean=0, std=0.02 / math.sqrt(2 * L))

        self.mask = torch.triu(torch.ones(S, S), diagonal=1).bool().view(1, 1, S, S)
        self.cos, self.sin = precompute_rope(cfg)

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
            ignore_index=-1
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
