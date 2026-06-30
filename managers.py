import math
import os
import gpustat
import wandb
import torch
import torch.distributed as dist
from collections import defaultdict, OrderedDict
from typing import Optional
from contextlib import contextmanager
from argparse import Namespace as ArgParseNamespace
from tqdm.auto import tqdm  # Use auto to support both terminal and Jupyter

from constants import (
    TEXT_LRS_ITER,

    TEXT_TRAIN_ITER,
    TEXT_VAL_ITER,
    TEXT_ELAPSED_ITER,
    TEXT_GPU_MEM_ITER,

    TEXT_GPU_MEM_MIN,
    TEXT_GPU_MEM_AVG,
    TEXT_GPU_MEM_MAX,
)

class TimingManager:
    def __init__(self):
        self.metrics = defaultdict(float)

    @contextmanager
    def record(self, block_name: str):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        yield # at this point __enter__ ends and python executes the code in the "with record('block')" clause

        # the code below runs after the code in with clause ends - this part of the function is equivalent to __exit__
        end_event.record()

        torch.cuda.synchronize()
        self.metrics[block_name] += end_event.elapsed_time(start_event)

    def get_then_reset(self):
        crt_metrics = self.metrics.copy()
        self.metrics.clear()
        return crt_metrics

class LoggingManager:
    def __init__(self, cfg: ArgParseNamespace):
        self.is_master = True
        if dist.is_initialized():
            self.is_master = dist.get_rank() == 0

        if self.is_master:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                group=cfg.wandb.group,
                job_type=cfg.wandb.job_type,
                name=cfg.wandb.run_name,
                config=cfg,
            )

        self.cfg = cfg

        self.metrics = {}
        self.train_pbar = None
        self.eval_pbar = None

    #####
    ##### Progress bar methods
    #####
    def start_train_pbar(self, total_iters: int):
        if self.is_master:
            self.train_pbar = tqdm(
                total=total_iters,
                desc="Training",
                dynamic_ncols=True,
                smoothing=0.01
            )
    def start_eval_pbar(self, total_batches: int):
        if self.is_master:
            self.eval_pbar = tqdm(
                total=total_batches,
                desc="Evaluation",
                leave=False, # disappears after completion
                dynamic_ncols=True
            )

    def update_eval_pbar(self, step_inc: int = 1, **kwargs):
        if self.is_master:
            self._update_pbar(self.eval_pbar, step_inc, **kwargs)

    def update_train_pbar(self, step_inc: int = 1, **kwargs):
        if self.is_master:
            self._update_pbar(self.train_pbar, step_inc, **kwargs)

    def _update_pbar(self, pbar, step_inc: int = 1, **kwargs):
        if self.is_master and pbar:
            pbar.update(step_inc)
            if kwargs:
                pbar.set_postfix(**kwargs)

    def close_eval_pbar(self):
        if self.is_master and self.eval_pbar:
            self.eval_pbar.close()
            self.eval_pbar = None

    #####
    ##### Different statistics
    #####
    @torch.no_grad()
    def log_model_info(self, model, world_size):
        if self.is_master:
            cfg = self.cfg
            is_tied = cfg.model.tie_embeds
            CTR = cfg.training.chinchilla_token_ratio
            S = cfg.model.seq_len
            B = cfg.training.local_batch_size
            A = cfg.training.grad_acc_steps

            params_inp_embed = model.tok_embed.weight.numel()
            params_pos_embed = model.pos_embed.weight.numel() if hasattr(model, "pos_embed") else 0
            params_out_embed = model.lm_head.weight.numel()
            params_linear = sum(p.numel() for p in model.blocks.parameters() if p.ndim == 2)
            params_layer_norm = sum(p.numel() for p in model.parameters() if p.ndim == 1)
            params_all = sum(p.numel() for p in model.parameters())


            T = params_linear * CTR * 20
            print("-" * 50)
            print("Model:")
            print(str(model))
            print("-" * 50)
            print(f"Model summary:")
            print(f"\t Input Embedding params: {params_inp_embed:_}") # (1 - int(is_tied)) * params_out_embed (tied: {is_tied})
            print(f"\t   Pos Embedding params: {params_pos_embed:_}")
            print(f"\tOutput Embedding params: {params_out_embed * (1 - int(is_tied)):_} (tied: {is_tied})")
            print(f"\t          Linear params: {params_linear:_}")
            print(f"\t     Layer norms params: {params_layer_norm:_}")
            print(f"\t             All params: {params_all:_}")
            print("-" * 50)
            print(f"Tokens info:")
            print(f"\t           Linear params (P): {params_linear:_}")
            print(f"\tChinchilla Token Ratio (CTR): {CTR * 20}x")
            print(f"\t        Tokens (T = P * CTR): {T:_}")
            print(f"\t         Sequence Length (S): {S}")
            print(f"\t       Device Batch Size (B): {B}")
            print(f"\t          Grad Acc Steps (A): {A}")
            print(f"\t              World Size (W): {world_size}")
            print(f"\t Steps (T / (B * S * A * W)): {math.ceil(T / (B * S * A * world_size))}")
            print("-" * 50)

    #####
    ##### Logging methods
    #####
    def log_memory_usage(self):
        if self.is_master:
            gpus = gpustat.new_query().gpus
            # gids = list(range(dist.get_world_size())) if ddp else list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
            gids = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
            mem_per_gpu = [
                int(proc['gpu_memory_usage']) / 1000.  # MB to GB
                for gid in gids
                for proc in gpus[gid]['processes']
            ]
            avg = round(sum(mem_per_gpu) / len(mem_per_gpu), 2)
            M = round(max(mem_per_gpu), 2)
            m = round(min(mem_per_gpu), 2)

            self.metrics[TEXT_GPU_MEM_MIN] = m
            self.metrics[TEXT_GPU_MEM_AVG] = avg
            self.metrics[TEXT_GPU_MEM_MAX] = M

    def log_redundant_iters(self, crt_iter):
        if self.is_master:
            self.metrics[TEXT_LRS_ITER] = crt_iter
            self.metrics[TEXT_TRAIN_ITER] = crt_iter
            self.metrics[TEXT_VAL_ITER] = crt_iter
            self.metrics[TEXT_ELAPSED_ITER] = crt_iter
            self.metrics[TEXT_GPU_MEM_ITER] = crt_iter

    def log(self, key, value):
        if self.is_master:
            self.metrics[key] = value

    def log_dict(self, other_metrics):
        if self.is_master:
            self.metrics.update(other_metrics)

    def log_console(self, message: str, data: Optional[OrderedDict] = None):
        if self.is_master:
            log_parts = []
            if data:
                log_parts.extend(f"{k}={v}" for k, v in data.items())
            print(f"{message}: " + " | ".join(log_parts))

    def commit(self, step: int):
        if self.is_master and self.metrics:
            wandb.log(self.metrics, step=step)
            self.metrics.clear()

    def finish(self):
        if self.is_master:
            wandb.finish()
