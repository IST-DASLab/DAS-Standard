import os
import gpustat
import wandb
import torch
import torch.distributed as dist
from collections import defaultdict
from contextlib import contextmanager
from argparse import Namespace as ArgParseNamespace

from constants import (
    TEXT_ITER,
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

        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            job_type=cfg.wandb.job_type,
            name=cfg.wandb.name,
            config=cfg,
        )

        self.metrics = {}

    def log_memory_usage(self):
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
        self.metrics[TEXT_ITER] = crt_iter
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

    def commit(self, step: int):
        if self.is_master and self.metrics:
            wandb.log(self.metrics, step=step)
            self.metrics.clear()

    def finish(self):
        if self.is_master:
            wandb.finish()
