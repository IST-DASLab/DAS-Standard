import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from argparse import Namespace as ArgParseNamespace

class LlamaFeedForward(nn.Module):
    def __init__(self, cfg: ArgParseNamespace):
        super().__init__()

        E = cfg.model.embed_size
        dtype = cfg.model.dtype

        self.fc1 = nn.Linear(E, E, dtype=dtype, bias=False)
        self.fc2 = nn.Linear(E, E, dtype=dtype, bias=False)
        self.fc3 = nn.Linear(E, E, dtype=dtype, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = self.silu(x1) * x2
        x = self.fc3(x)
        return x


class LlamaBlock(nn.Module):
    def __init__(self, cfg: ArgParseNamespace, layer_id: int):
        super().__init__()


class Llama2(nn.Module):
    def __init__(self, cfg: ArgParseNamespace):
        super().__init__()
        self.cfg = cfg

        self.tok_embed = nn.Embedding(cfg.model.vocab_size, cfg.model.embed_size)
        self.blocks = nn.ModuleList([
            LlamaBlock(cfg, layer_id)
            for layer_id in range(cfg.model.num_layers)
        ])