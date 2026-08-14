import sys
import inspect

import torch
from simple_parsing import ArgumentParser
from argparse import Namespace as ArgParseNamespace
from dataclasses import dataclass, is_dataclass
from typing import Literal

from config import model_config_factory
from config.utils import _compute_max_iters
from models.attention import MultiHeadSelfAttention, GroupedQueryAttention
from models.feedforward import GPTFeedForward, LlamaFeedForward


# @dataclass
# class UserConfig:
#     model_name: Literal["gpt", "llama", "olmo"] = "llama"
#     model_dtype: Literal["bf16", "fp16", "fp32"] = "fp32"
#     model_attn_type: Literal["mhsa", "mhsa-flash", "gqa", "gqa-flash"] = "mhsa"
#     model_custom_size: str = "100M"
#     model_seq_len: int = 1024
#     training_tokens_count: float = 0 # zero means disabled

@dataclass
class ModelConfig:
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
class DatasetConfig:
    dataset_path: str = "./data"
    dataset_name: Literal["c4"] = "c4"


@dataclass
class TrainingConfig: # mainly set by user
    model_name: Literal["gpt", "llama2", "llama3", "olmo" ] = "llama2"
    model_size: str = "100M"
    seed: int = 42
    tokens_count: float = 0 # zero means disabled (lower priority than `chinchilla_token_ratio`)
    chinchilla_token_ratio: int = 1 # how many chinchillas to train the model for (higher priorith than `tokens_count`)
    local_batch_size: int = 64 # per device batch size
    grad_acc_steps: int = 8
    max_iters: int = -1 # will be automatically determined based on number of tokens, seq len, device batch size and number of workers
    backend: Literal["none", "ddp", "fsdp2"] = "ddp"
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    compile_model: bool = False
    eval_interval: int = 200
    eval_batches: int = 32
    output_folder: str = "./output"


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
    muon_lr: float = 1e-3
    muon_momentum: float = 0.95


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
    """
    parser = ArgumentParser(description="Arguments are read from the dataclass fields *Config in configs package, such as ModelConfig, OptimizerConfig, etc.")

    for name, obj in inspect.getmembers(sys.modules[__name__]): # current module
        if inspect.isclass(obj) and is_dataclass(obj) and "config" in name.lower():
            dest_name = name.lower().replace("config", "")
            parser.add_arguments(obj, dest=dest_name)
            # print(f'Found config "{name}": all fields are available in the "{dest_name}" object')

    return preprocess(parser.parse_args())


def preprocess(cfg: ArgParseNamespace):
    attn_params_fn = None
    mlp_params_fn = None
    model_name = cfg.training.model_name
    model_size = cfg.training.model_size

    #####
    ##### Use predefined model Config
    #####
    cfg.model = model_config_factory(model_name, model_size)

    #####
    ##### Set functions to compute params count
    #####
    if "gpt" in model_name:
        attn_params_fn = MultiHeadSelfAttention.get_param_count
        mlp_params_fn = GPTFeedForward.get_param_count
    elif "llama" in model_name:
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
    cfg.training.max_iters = _compute_max_iters(cfg, attn_fn=attn_params_fn, mlp_fn=mlp_params_fn)

    assert cfg.model.embed_size % cfg.model.num_heads == 0

    #####
    ##### Map model dtype from string to torch dtype;
    #####
    match cfg.model.dtype:
        case "bf16":
            cfg.model.dtype = torch.bfloat16
        case "fp16":
            cfg.model.dtype = torch.float16
        case "fp32":
            cfg.model.dtype = torch.float32

    return cfg