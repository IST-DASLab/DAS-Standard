from argparse import Namespace as ArgParseNamespace

from models.gpt import GPT


def get_model(args: ArgParseNamespace):
    match args.model.model_name:
        case "gpt":
            return GPT(args)
        case _:
            raise NotImplementedError(f"Model {args.model.model_name} not implemented")
