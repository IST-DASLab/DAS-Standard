import os
import torch
import random
import numpy as np
from argparse import Namespace as ArgParseNamespace
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


def get_backend(cfg: ArgParseNamespace):
    match cfg.training.backend:
        case "single":
            return SingleBackend(cfg)
        case "ddp":
            return DDPBackend(cfg)
        case "fsdp2":
            return FSDP2Backend(cfg)
        case _:
            raise NotImplementedError(f"Backend {cfg.training.backend} is not implemented")

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
            }[self.cfg.training.amp_dtype]
        )

    @property
    def is_master(self):
        return self.rank == 0

    def terminate(self):
        pass # do nothing by default

    def get_device(self):
        return f"cuda:{self.local_rank}"

    def set_grad_sync(self, model, enabled: bool):
        pass

    def wrap_model_then_compile(self, model):
        # no wrapping in the base function
        if self.cfg.training.compile_model:
            return torch.compile(model.to(self.device))
        return model.to(self.device)

    def clip_grad_norm_(self, model, max_norm: float):
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm if max_norm > 0 else float("inf"))
        if isinstance(total_norm, DTensor):
            total_norm = total_norm.full_tensor()
        return total_norm.item()


class SingleBackend(BaseBackend):
    def __init__(self, cfg: ArgParseNamespace):
        super().__init__(
            cfg=cfg,
            rank=0,
            local_rank=0,
            world_size=1
        )
        if self.is_master:
            print(f"\n=== Using Single GPU backend ===\n")

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

class FSDP2Backend(BaseBackend):
    def __init__(self, cfg: ArgParseNamespace):
        dist.init_process_group(backend="nccl")
        super().__init__(
            cfg=cfg,
            rank=int(os.environ["RANK"]),
            local_rank=int(os.environ["LOCAL_RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )

        self.mesh = init_device_mesh(device_type="cuda", mesh_shape=(self.world_size, ))
        self.cfg.training.grad_acc_steps //= self.world_size # same as in DDP

        if self.is_master:
            print(f"\n=== Using FSDP2 backend ===\n")

    def wrap_model_then_compile(self, model):
        model = model.to(self.device) # TODO: use meta-device for large models

        policy = MixedPrecisionPolicy(param_dtype=self.cfg.model.dtype, reduce_dtype=torch.float32)
        for block in model.blocks:
            fully_shard(block, mesh=self.mesh, mp_policy=policy)
        fully_shard(model, mesh=self.mesh, mp_policy=policy)

        if self.cfg.training.compile_model:
            model = torch.compile(model)

        return model

    def get_amp_context(self):
        """
        MixedPrecisionPolicy used in `wrap_model_then_compile` already runs compute in bf16 and reduces grads in fp32, so autocast is redundant.
        """
        return nullcontext()

    def set_grad_sync(self, model, enabled: bool):
        model.set_requires_gradient_sync(enabled) # this is the equivalent of no_sync in FSDP2

    def terminate(self):
        dist.destroy_process_group()
