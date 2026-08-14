from argparse import Namespace as ArgParseNamespace

from models.gpt import GPT
from models.llama import Llama
from models.qwen import Qwen3


def get_model(cfg: ArgParseNamespace):
    match cfg.training.model_name:
        case "gpt":
            return GPT(cfg)
        case "llama2" | "llama3":
            return Llama(cfg)
        case "qwen3":
            return Qwen3(cfg)
        case _:
            raise NotImplementedError
