import torch

from .constants import *
from .evaluation import eval_and_log

def get_dtype(name):
    match name:
        case "bf16":
            return torch.bfloat16
        # case "fp16":
        #     return torch.float16
        case "fp32":
            return torch.float32
        case _:
            raise RuntimeError(f"Unknown dtype '{name}', only 'bf16', 'fp16' and 'fp32' are currently supported")
