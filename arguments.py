import inspect
from dataclasses import is_dataclass
from simple_parsing import ArgumentParser
import configs

def get_dynamic_arg_parser():
    """
    This function iterates through all dataclasses defined in the config.py file
    and creates command line arguments dynamically.

    After parsing the args, they are preprocessed before returning them in the
    main process for usage.
    """
    parser = ArgumentParser(description="Arguments are read from the dataclass fields *Config in configs package, such as ModelConfig, OptimizerConfig, etc.")

    for name, obj in inspect.getmembers(configs):
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
        case "llama":
            args.model.vocab_size = 32_000
        case "olmo":
            raise NotImplementedError
            # args.model.vocab_size = None

    """
        OptimizerConfig ==> optimizer
    """


    return args