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

def round_param_count(count):
    if count >= 1_000_000_000:
        # round to 100M for models larger than 1B
        quantum = 100_000_000
    else:
        # round to 1M for models smaller than 1B
        quantum = 1_000_000
    return ((count + quantum - 1) // quantum) * quantum

def count_transformer_block_params(model, vocab_size):
    """
    This functions returns all parameters from the transformer blocks by
    excluding the embedding layers.
    """
    count = 0
    for p in model.parameters():
        is_embedding = p.ndim == 2 and vocab_size in p.shape
        if not is_embedding:
            count += p.numel()
    return count