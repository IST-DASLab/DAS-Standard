import sys
import inspect
from simple_parsing import ArgumentParser
from dataclasses import dataclass, is_dataclass
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
    weight_tying: bool = None
    mlp_expansion_factor: int = 4

def get_dynamic_arg_parser():
    """
    This function iterates through all dataclasses defined in the config.py file
    and creates command line arguments dynamically.

    After parsing the args, they are preprocessed before returning them in the
    main process for usage.
    """
    parser = ArgumentParser(description="Arguments are read from the dataclass fields *Config in configs package, such as ModelConfig, OptimizerConfig, etc.")

    for name, obj in inspect.getmembers(sys.modules[__name__]): # current module
        if inspect.isclass(obj) and is_dataclass(obj) and "config" in name.lower():
            dest_name = name.lower().replace("config", "")
            parser.add_arguments(obj, dest=dest_name)
            print(f'Found config "{name}": all fields are available in the "{dest_name}" object')

    return preprocess(parser.parse_args())

def preprocess(args):
    """
        ModelConfig ==> model
    """
    match args.model.name:
        case "gpt":
            args.model.vocab_size = 50_304 # GPT-2 has vocab_size=50_257, padded up to nearest multiple of 64
            if args.model.weight_tying is None:
                args.model.weight_tying = True # the new default
        case "llama":
            args.model.vocab_size = 32_000
        case "olmo":
            raise NotImplementedError
            # args.model.vocab_size = None

    assert args.model.embed_size % args.model.num_heads == 0
    """
        OptimizerConfig ==> optimizer
    """

    return args