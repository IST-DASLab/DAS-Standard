import torch
from argparse import Namespace as ArgParseNamespace

from models.gpt import GPT
from models.llama import Llama
from models.qwen import Qwen3


def get_model(cfg: ArgParseNamespace):
    model_name_size = cfg.training.model_name_size

    if "gpt" in model_name_size:
        model = GPT(cfg)
    elif "llama" in model_name_size:
        model = Llama(cfg)
    elif "qwen3" in model_name_size:
        model = Qwen3(cfg)
    else:
        raise NotImplementedError

    return model.to(dtype=dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[cfg.model.dtype])
