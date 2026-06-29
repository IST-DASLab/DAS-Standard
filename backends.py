import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import numpy as np
import torch.distributed as dist
from argparse import Namespace as ArgParseNamespace
from contextlib import nullcontext
from tqdm import tqdm

def get_backend(args: ArgParseNamespace):
    match args.training.backend:
        case "ddp":
            return DDPBackend(args)
        case _:
            raise NotImplementedError(f"Backend {args.training.backend} is not implemented")

class BaseBackend:
    def __init__(self, cfg: ArgParseNamespace, rank, local_rank, world_size):
        self.cfg = cfg
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size

        self.device = f'cuda:{self.local_rank}'

        self.general_setup()

    def general_setup(self):
        seed = self.cfg.training.seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        torch.cuda.set_device(self.device)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def get_amp_context(self):
        return torch.amp.autocast(
            device_type="cuda",
            dtype={
                'fp32': torch.float32,
                'bf16': torch.bfloat16,
                'fp16': torch.float16,
            }[self.cfg.training.dtype]
        )

    def terminate(self):
        pass # do nothing by default

    @property
    def is_master(self):
        return self.rank == 0

    def wrap_model(self, model):
        if self.cfg.training.compile_model:
            return torch.compile(model.to(self.device))
        return model.to(self.device)


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
            For DDP, the grad acc_steps will be changed form 64 to 8 because we
            have 8 processes running in parallel
        """
        self.cfg.training.grad_acc_steps //= self.world_size

    def wrap_model(self, model):
        wrapped = DDP(model, device_ids=[self.local_rank]).to(self.device)
        if self.cfg.training.compile_model:
            return torch.compile(wrapped)
        return wrapped

    def terminate(self):
        dist.destroy_process_group()


class NoneBackend(BaseBackend):
    def __init__(self, cfg: ArgParseNamespace):
        super().__init__(
            cfg=cfg,
            rank=0,
            local_rank=0,
            world_size=1
        )