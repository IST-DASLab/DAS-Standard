import torch
import torch.nn as nn
from torch.nn import functional as F
from argparse import Namespace as ArgParseNamespace

from models.attention import attn_factory
from models.feedforward import LlamaFeedForward
from models.norm import RMSNorm
from models.rope import precompute_rope


class LlamaBlock(nn.Module):
    def __init__(self, cfg: ArgParseNamespace):
        super().__init__()
        E = cfg.model.embed_size
        self.attn = attn_factory(cfg)
        self.mlp = LlamaFeedForward(cfg)
        self.norm1 = RMSNorm(ndim=E)
        self.norm2 = RMSNorm(ndim=E)

    def forward(self, x, mask, cos, sin):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, mask, cos, sin)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + shortcut

        return x


class Llama(nn.Module):
    def __init__(self, cfg: ArgParseNamespace):
        super().__init__()
        self.cfg = cfg

        E = cfg.model.embed_size
        V = cfg.model.vocab_size
        L = cfg.model.num_layers
        S = cfg.model.seq_len
        dtype = cfg.model.dtype

        assert not cfg.model.bias, "Llama model needs bias=False, but found bias=True"

        self.tok_embed = nn.Embedding(V, E, dtype=dtype)
        self.blocks = nn.ModuleList([LlamaBlock(cfg) for _ in range(L)])
        self.norm = RMSNorm(ndim=E)
        self.lm_head = nn.Linear(in_features=E, out_features=V, bias=False, dtype=dtype)

        self.cos, self.sin = precompute_rope(cfg)
        self.mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool),
            diagonal=1
        ).view(1, 1, S, S)

    def forward(self, inputs, targets):
        dtype = self.cfg.model.dtype

        x = self.tok_embed(inputs)
        for block in self.blocks:
            x = block(x, self.mask, self.cos, self.sin)
        x = self.norm(x)
        logits = self.lm_head(x).to(dtype=dtype)

        loss = F.cross_entropy(
            input=logits.view(-1, logits.size(-1)),
            target=targets.view(-1),
            ignore_index=-1
        )
        return loss
