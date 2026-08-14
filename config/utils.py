import math

from .ladder_gpt import *
from .ladder_llama2 import *
from .ladder_llama3 import *

def model_config_factory(model_name, model_size):
    key = f"{model_name.lower()}_{model_size.lower()}"
    return {
        "gpt_3m": config_gpt_3m,
        "gpt_11m": config_gpt_11m,
        "gpt_25m": config_gpt_25m,
        "gpt_49m": config_gpt_49m,
        "gpt_85m": config_gpt_85m,
        "gpt_200m": config_gpt_200m,
        "gpt_300m": config_gpt_300m,
        "gpt_710m": config_gpt_710m,
        "gpt_1.5b": config_gpt_1500m,
        "gpt_2.5b": config_gpt_2500m,
        "gpt_6.4b": config_gpt_6400m,
        "gpt_12.6b": config_gpt_12600m,
        "llama2_35m": config_llama2_35m,
        "llama2_60m": config_llama2_60m,
        "llama2_135m": config_llama2_135m,
        "llama2_250m": config_llama2_250m,
        "llama2_700m": config_llama2_700m,
        "llama2_1.1b": config_llama2_1100m,
        "llama2_2.8b": config_llama2_2800m,
        "llama2_7b": config_llama2_7b,
        "llama2_13b": config_llama2_13b,
        "llama2_34b": config_llama2_34b,
        "llama2_70b": config_llama2_70b,
        "llama3_90m": config_llama3_90m,
        "llama3_200m": config_llama3_200m,
        "llama3_350m": config_llama3_350m,
        "llama3_1b": config_llama3_1b,
        "llama3_3b": config_llama3_3b,
        "llama3_8b": config_llama3_8b,
        "llama3_17b": config_llama3_17b,
        "llama3_70b": config_llama3_70b,
        "llama3_405b": config_llama3_405b,
    }[key]()

def _compute_max_iters(cfg, attn_fn, mlp_fn):
    """
    This function computes the number of iterations based on some number of
    parameters for the model (can be full model size of
    chinchilla token ratio and batch size / grad acc steps / seq len.

    We only count the number of linear layers in transformer blocks (Attn & MLP)
    """
    L = cfg.model.num_layers
    S = cfg.model.seq_len
    B = cfg.training.local_batch_size
    A = cfg.training.grad_acc_steps
    CTR = cfg.training.chinchilla_token_ratio

    params_count = L * (attn_fn(cfg) + mlp_fn(cfg))
    num_tokens = params_count * CTR * 20
    iterations = math.ceil(num_tokens / (B * S * A))

    return iterations
