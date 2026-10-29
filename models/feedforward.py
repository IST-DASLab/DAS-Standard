from argparse import Namespace as ArgParseNamespace
import torch
from torch import nn as nn


class GPTFeedForward(nn.Module):
    # @staticmethod
    # def get_param_count(cfg: ArgParseNamespace):
    #     E = cfg.model.embed_size
    #     exp_factor = cfg.model.gpt_mlp_exp_factor
    #     return 2 * E * (E * exp_factor)

    def __init__(self, cfg: ArgParseNamespace):
        super().__init__()
        self.cfg = cfg

        E = cfg.model.embed_size
        E_expanded = E * cfg.model.gpt_mlp_exp_factor
        bias = cfg.model.bias

        self.fc = nn.Linear(in_features=E, out_features=E_expanded, bias=bias)
        self.Wo = nn.Linear(in_features=E_expanded, out_features=E, bias=bias)
        # self.dropout = nn.Dropout(cfg.model.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = self.Wo(x)
        # x = self.dropout(x)
        return x


class LlamaFeedForward(nn.Module):
    # @staticmethod
    # def get_param_count(cfg: ArgParseNamespace):
    #     E = cfg.model.embed_size
    #     Hdim = cfg.model.llama_mlp_size
    #     return 3 * E * Hdim

    def __init__(self, cfg: ArgParseNamespace):
        super().__init__()

        E = cfg.model.embed_size
        Hdim = cfg.model.llama_mlp_size
        dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[cfg.model.dtype]

        self.fc1 = nn.Linear(in_features=E, out_features=Hdim, dtype=dtype, bias=False)
        self.fc2 = nn.Linear(in_features=E, out_features=Hdim, dtype=dtype, bias=False)
        self.fc3 = nn.Linear(in_features=Hdim, out_features=E, dtype=dtype, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = self.silu(x1) * x2
        x = self.fc3(x)
        return x
