import math
import os
from argparse import Namespace as ArgParseNamespace
from collections import OrderedDict
from typing import Optional

import gpustat
import torch
import wandb
from torch import distributed as dist
from tqdm.asyncio import tqdm

from config import print_config
from models import round_param_count
from utils import *


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
            wandb.define_metric("*", step_metric=TEXT_XAXIS_ITER)

        self.cfg = cfg
        self.metrics = {}
        self.train_pbar = None
        self.eval_pbar = None

    #####
    ##### Progress bar methods
    #####
    def start_train_pbar(self, start_iter: int, total_iters: int):
        if self.is_master:
            self.train_pbar = tqdm(
                initial=start_iter,
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
    def log_config_and_model(self, model, world_size):
        if self.is_master:
            cfg = self.cfg
            is_tied = cfg.model.tie_embeds
            CTR = cfg.training.chinchilla_token_ratio
            S = cfg.model.seq_len
            B = cfg.training.local_batch_size
            A = cfg.training.grad_acc_steps

            print_config(cfg)

            PIE = model.token_embed.weight.numel()
            PPE = model.pos_embed.weight.numel() if hasattr(model, "pos_embed") else 0
            POE = 0 if is_tied else model.lm_head.weight.numel()
            PL = sum(p.numel() for p in model.blocks.parameters() if p.ndim == 2)
            PN = sum(p.numel() for p in model.parameters() if p.ndim == 1)
            PA = sum(p.numel() for p in model.parameters())

            TP = PL + PN
            RTP = round_param_count(TP)
            T = RTP * CTR * 20

            print("Model:")
            print(str(model))
            print("=" * 50)
            print(f"Model summary:")
            print(f"\t                Input Embedding params (PIE): {PIE:_}")
            if PPE > 0:
                print(f"\t                  Pos Embedding params (PPE): {PPE:_}")
            print(f"\t               Output Embedding params (POE): {POE:_} (tied: {is_tied})")
            print(f"\t                          Linear params (PL): {PL:_}")
            print(f"\t                            Norm params (PN): {PN:_}")
            print(f"\t All params (PIE + PPE + POE + PL + PN = PA): {PA:_}")

            print("=" * 50)

            print(f"Total tokens & training steps:")
            print(f"\t     Token Params (PL + PN = TP): {TP:_}")
            print(f"\t      Rounded Token Params (RTP): {RTP:_}")
            print(f"\t    Chinchilla Token Ratio (CTR): {CTR * 20}x")
            print(f"\t          Tokens (CTR * RTP = T): {T:_}")
            print(f"\t             Sequence Length (S): {S}")
            print(f"\t           Device Batch Size (B): {B}")
            print(f"\t              Grad Acc Steps (A): {A}")
            print(f"\t                  World Size (W): {world_size}")
            print(f"\t     Steps (T / (B * S * A * W)): {math.ceil(T / (B * S * A * world_size)):_}")

            print("=" * 50)

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

    def log_xaxis(self, iter, tokens):
        if self.is_master:
            # log traces that can be used as x-axis in any panel grid
            self.metrics[TEXT_XAXIS_ITER] = iter
            self.metrics[TEXT_XAXIS_TOKENS] = tokens
            self.metrics[TEXT_XAXIS_PROG_PRCT] = round(iter / self.cfg.training.max_iters, 4) * 100 # percentage between 0 and 100%

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

    def log_all_stats(self, crt_iter, total_step_loss, grad_norm, timer, optimizer):
        """
        This function logs all required information at the end of the training loop
        :param crt_iter: current training step
        :param total_step_loss: total training loss at the current training step
        :param grad_norm: norm of the gradient at the current training step
        :param timer: the timer object containing all time measurements for one training step
        :param optimizer: the MultiOptimizer
        """
        W = dist.get_world_size() if dist.is_initialized() else 1

        self.update_train_pbar(step_inc=1, loss=f"{total_step_loss:.4f}")

        self.log_dict({
            TEXT_TRAIN_LOSS: total_step_loss,
            TEXT_TRAIN_PPL: math.exp(total_step_loss),
            TEXT_TRAIN_GRAD_NORM: grad_norm,
        })
        self.log_dict(timer.get_then_reset())
        self.log_dict(optimizer.extract_learning_rates())
        self.log_dict(optimizer.get_optimizer_stats())
        self.log_memory_usage()
        self.log_xaxis(
            iter=crt_iter,
            tokens=crt_iter * self.cfg.model.seq_len * self.cfg.training.local_batch_size * self.cfg.training.grad_acc_steps * W,
        ) # log these values to all panel grids (section) in wandb
        self.commit(step=crt_iter)

    def commit(self, step: int):
        if self.is_master and self.metrics:
            wandb.log(self.metrics, step=step)
            self.metrics.clear()

    def finish(self):
        if self.is_master:
            wandb.finish()
