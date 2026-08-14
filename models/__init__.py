from argparse import Namespace as ArgParseNamespace

from models.gpt import GPT
from models.llama import Llama
from models.qwen import Qwen3


def get_model(cfg: ArgParseNamespace):
    model_name_size = cfg.training.model_name_size
    if "gpt" in model_name_size:
        return GPT(cfg)
    if "llama" in model_name_size:
        return Llama(cfg)
    if "qwen3" in model_name_size:
        return Qwen3(cfg)
    raise NotImplementedError
