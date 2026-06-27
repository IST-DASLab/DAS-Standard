from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelConfig:
    name: Literal["gpt", "llama", "olmo"] = "llama"
    vocab_size: int = -1
    num_layers: int = 6
    num_heads: int = 8
    embed_size: int = 1024
    seq_len: int = 1024
    dropout: float = 0.0 # see GPT class
    bias: bool = False
    weight_tying: bool = False
