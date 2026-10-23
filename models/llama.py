import torch
import torch.nn as nn
from torch.nn import functional as F
from argparse import Namespace as ArgParseNamespace

from models.attention import Attention # , attn_factory
from models.feedforward import LlamaFeedForward
from models.norm import RMSNorm
from models.rope import precompute_rope


class LlamaBlock(nn.Module):
    def __init__(self, cfg: ArgParseNamespace):
        super().__init__()
        E = cfg.model.embed_size
        self.attn = Attention(cfg)
        self.mlp = LlamaFeedForward(cfg)
        self.norm1 = RMSNorm(ndim=E, bias=False)
        self.norm2 = RMSNorm(ndim=E, bias=False)

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
        dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[cfg.model.dtype]

        assert not cfg.model.bias, "Llama model needs bias=False, but found bias=True"

        self.token_embed = nn.Embedding(V, E, dtype=dtype)
        self.blocks = nn.ModuleList([LlamaBlock(cfg) for _ in range(L)])
        self.norm = RMSNorm(ndim=E)
        self.lm_head = nn.Linear(in_features=E, out_features=V, bias=False, dtype=dtype)

        if cfg.model.tie_embeds:
            self.token_embed.weight = self.lm_head.weight

        cos, sin = precompute_rope(cfg)
        self.register_buffer(name="cos", tensor=cos, persistent=False)
        self.register_buffer(name="sin", tensor=sin, persistent=False)
        self.register_buffer(
            name="mask",
            tensor=torch.triu(torch.ones(S, S), diagonal=1).bool().view(1, 1, S, S),
            persistent=False)

    def forward(self, inputs, targets):
        dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[self.cfg.model.dtype]

        x = self.token_embed(inputs)
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
