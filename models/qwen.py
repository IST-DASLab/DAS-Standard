from argparse import Namespace as ArgParseNamespace

from models import Llama

class Qwen3(Llama):
    """
    The difference between Qwen3 and Llama2/3 is in config:
    - Qwen3 has QK-norm, Llama2/3 doesn't
    - Qwen3 has different sizes for embedding, MLP, num layers, num heads
    Therefore, we just inherit everything from Llama
    """
    def __init__(self, cfg: ArgParseNamespace):
        super().__init__(cfg)

    def forward(self, x):
        return super().forward(x)
