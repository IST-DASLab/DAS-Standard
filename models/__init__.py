from argparse import Namespace as ArgParseNamespace

from models.gpt import GPT


def get_model(cfg: ArgParseNamespace):
    match cfg.model.model_name:
        case "gpt":
            return GPT(cfg)
        case _:
            raise NotImplementedError(f"Model {cfg.model.model_name} not implemented")
