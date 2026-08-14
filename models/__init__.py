from argparse import Namespace as ArgParseNamespace

from models.gpt import GPT
from models.llama import Llama


def get_model(cfg: ArgParseNamespace):
    match cfg.training.model_name:
        case "gpt":
            return GPT(cfg)
        case "llama2" | "llama3":
            return Llama(cfg)
        case _:
            raise NotImplementedError
