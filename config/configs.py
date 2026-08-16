import sys
import torch
import inspect
from typing import Literal
from simple_parsing import ArgumentParser
from dataclasses import dataclass, is_dataclass
from argparse import Namespace as ArgParseNamespace

from models.attention import MultiHeadSelfAttention, GroupedQueryAttention
from models.feedforward import GPTFeedForward, LlamaFeedForward


@dataclass
class TrainingConfig: # mainly set by user
    model_name_size: str = "llama3-1b"
    seed: int = 42
    tokens_count: float = 0 # zero means disabled (lower priority than `chinchilla_token_ratio`)
    chinchilla_token_ratio: int = 1 # how many chinchillas to train the model for (higher priorith than `tokens_count`)
    local_batch_size: int = 64 # per device batch size
    grad_acc_steps: int = 8
    max_iters: int = -1 # will be automatically determined based on number of tokens, seq len, device batch size and number of workers
    backend: Literal["single", "ddp", "fsdp2"] = "ddp"
    amp_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    compile_model: bool = False
    eval_interval: int = 200
    eval_batches: int = 32
    output_folder: str = "./output"


@dataclass
class ModelConfig: # initialized via model ladders
    dtype: Literal["bf16", "fp16", "fp32"] = "fp32"
    attn_type: Literal["mhsa", "mhsa-flash", "gqa", "gqa-flash"] = "mhsa"
    attn_gqa_kvg: int = 8
    qk_norm: bool = False
    rope_theta: float = 10_000.0
    vocab_size: int = -1
    num_layers: int = 6
    num_heads: int = 8
    embed_size: int = 1024
    seq_len: int = 1024
    dropout: float = 0.0
    bias: bool = False
    tie_embeds: bool = False
    gpt_mlp_exp_factor: int = 4
    llama_mlp_size: int = 11_008


@dataclass
class QATConfig:
    method_name: Literal["quartet2", "disabled"] = "disabled"
    hadamard_size: int = 128


@dataclass
class OptimizerConfig:
    optimizer_name: Literal["adamw", "muon"] = "adamw"
    weight_decay: float = 0.0

    grad_clipping: float = 1.0 # rescale the global gradient to have this norm. Set to 0 to disable

    # Learning Rate scheduling
    lr_sched_name: Literal["cosine", "step"] = "cosine"
    lr_sched_warmup: float = 0.1 # percentage of number of steps at the beginning of training, e.g. first 10% steps we do linear lr warmup
    lr_sched_start_ratio: float = 0.01 # for warmup, start the lr value from lr * lr_sched_start_ratio
    lr_sched_end_ratio: float = 0.1 # for the decay part (after warmup), decay the lr value from lr to lr * lr_sched_end_ratio

    # AdamW parameters
    adamw_lr: float = 1e-3
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    adamw_eps: float = 1e-8

    # Muon parameters
    muon_lr: float = 1e-2
    muon_momentum: float = 0.95


@dataclass
class DatasetConfig:
    dataset_path: str = "./data"
    dataset_name: Literal["c4"] = "c4"


@dataclass
class WandbConfig:
    entity: str = None
    project: str = None
    group: str = None
    job_type: str = None
    run_name: str = None


def get_dynamic_arg_parser():
    """
    This function iterates through all dataclasses defined in the config.py file
    and creates command line arguments dynamically.

    After parsing the cfg, they are preprocessed before returning them in the
    main process for usage.

    !!!!! Trick !!!!!
    To allow the user to override certain fields of the ModelConfig, we do the following:
    1) we parse only the training arguments to read model_name_size
    2) we build the full parser with dynamic defaults: for the model, we inject the model ladder as the base default and simple_parsing will apply the user
    overrides over these exact values from the ladder.

    For example, this is useful when we want to use llama2-1B with a smaller sequence length: the default sequence length will be 4096, but the user can
    override this field with 1024 by specifying --seq_len=1024 in the command line
    """
    from .utils import model_config_factory, print_config

    ##### step 1: build a temporary parser and extract `model_name_size`
    temp_parser = ArgumentParser(add_help=False)
    temp_parser.add_arguments(TrainingConfig, dest="training")
    temp_known_args, _ = temp_parser.parse_known_args() # parse only args from TrainingConfig (added, so it's known)
    model_name_size = temp_known_args.training.model_name_size

    ##### step 2: build the full parser now
    full_parser = ArgumentParser(description="Arguments are read from the dataclass fields *Config in configs package, such as ModelConfig, OptimizerConfig, etc.")

    for name, obj in inspect.getmembers(sys.modules[__name__]): # current module
        if inspect.isclass(obj) and is_dataclass(obj) and "config" in name.lower():
            dest_name = name.lower().replace("config", "")

            if dest_name == "model":
                model_ladder = model_config_factory(model_name_size)
                full_parser.add_arguments(obj, dest=dest_name, default=model_ladder)
            else:
                full_parser.add_arguments(obj, dest=dest_name)
            # print(f'Found config "{name}": all fields are available in the "{dest_name}" object')

    cfg = preprocess(full_parser.parse_args())
    print_config(cfg)
    return cfg


def preprocess(cfg: ArgParseNamespace):
    from .utils import compute_max_iters
    attn_params_fn = None
    mlp_params_fn = None
    model_name_size = cfg.training.model_name_size

    #####
    ##### Set functions to compute params count
    #####
    if "gpt" in model_name_size:
        attn_params_fn = MultiHeadSelfAttention.get_param_count
        mlp_params_fn = GPTFeedForward.get_param_count
    elif ("llama" in model_name_size) or ("qwen" in model_name_size):
        if "mhsa" in cfg.model.attn_type:
            attn_params_fn = MultiHeadSelfAttention.get_param_count
        elif "gqa" in cfg.model.attn_type:
            attn_params_fn = GroupedQueryAttention.get_param_count
        else:
            raise RuntimeError(f"Invalid attn_type for attn_params_fn!")
        mlp_params_fn = LlamaFeedForward.get_param_count

    #####
    ##### Compute number of iterations based on model size (only Linears from transformer blocks)
    #####
    cfg.training.max_iters = compute_max_iters(cfg, attn_fn=attn_params_fn, mlp_fn=mlp_params_fn)

    #####
    ##### Check model size
    #####
    assert cfg.model.embed_size % cfg.model.num_heads == 0

    #####
    ##### Map model dtype from string to torch dtype;
    #####
    cfg.model.dtype = { "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, }[cfg.model.dtype]

    return cfg