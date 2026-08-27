import os
from argparse import Namespace as ArgParseNamespace

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from backends.base_backend import BaseBackend


class DDPBackend(BaseBackend):
    def __init__(self, cfg: ArgParseNamespace):
        dist.init_process_group(backend="nccl")
        super().__init__(
            cfg=cfg,
            rank=int(os.environ["RANK"]),
            local_rank=int(os.environ["LOCAL_RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )

        """
            Settings:
            - world_size = 8
            - grad_acc_steps = 64
            For DDP, the grad acc_steps will be changed form 64 to 8 because we have 8 processes running in parallel
        """
        self.cfg.training.grad_acc_steps //= self.world_size
        if self.is_master:
            print(f"\n=== Using DDP backend ===\n")

    def wrap_model_then_compile(self, model):
        wrapped = DDP(model.to(self.device), device_ids=[self.local_rank])

        if self.cfg.training.compile_model:
            return torch.compile(wrapped)
        return wrapped

    def terminate(self):
        dist.destroy_process_group()

    def set_grad_sync(self, model, enabled: bool):
        model.require_backward_grad_sync = enabled
