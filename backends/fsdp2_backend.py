import os
from argparse import Namespace as ArgParseNamespace
from contextlib import nullcontext

import torch
from torch import distributed as dist
from torch.distributed.tensor import Shard
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from backends.base_backend import BaseBackend
from utils import get_dtype


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

        assert self.cfg.training.grad_acc_steps % self.world_size == 0
        self.cfg.training.grad_acc_steps //= self.world_size # same as in DDP

        if self.is_master:
            print(f"\n=== Using FSDP2 backend ===\n")

    def wrap_model_then_compile(self, model):
        def shard_by_largest_dim(p):
            if p.ndim == 2:
                R, C = p.shape
                if R >= C:
                    return Shard(0)
                return Shard(1)
            return None

        model = model.to(self.device) # TODO: use meta-device for large models

        policy = MixedPrecisionPolicy(
            # param_dtype specifies the dtype for the unsharded parameter and hence the dtype for forward/backward computation and the parameter all-gather.
            param_dtype=get_dtype(self.cfg.precision.compute_dtype),

            # reduce_dtype specifies the dtype for gradient reduction (i.e. reduce-scatter or all-reduce)
            reduce_dtype=get_dtype(self.cfg.precision.fsdp_reduce_dtype),
        )

        for block in model.blocks:
            fully_shard(block, mesh=self.mesh, mp_policy=policy, shard_placement_fn=shard_by_largest_dim)

        fully_shard(model, mesh=self.mesh, mp_policy=policy, shard_placement_fn=shard_by_largest_dim)

        if self.cfg.training.compile_model:
            model = torch.compile(model)

        return model

    def get_amp_context(self):
        """
        This is a no-op for the FSDP2 backend because the MixedPrecisionPolicy governs the dtypes:
        1) param_dtype: dtype for forward/backward and parameters all-gather
        2) reduce_dtype: dtype for gradient reduction (reduce-scatter or all-reduce)
        MixedPrecisionPolicy used in `wrap_model_then_compile` already runs compute in bf16 and reduces grads in fp32, so autocast is redundant.
        """
        return nullcontext()

    def set_grad_sync(self, model, enabled: bool):
        model.set_requires_gradient_sync(enabled) # this is the equivalent of no_sync in FSDP2

    def terminate(self):
        dist.destroy_process_group()
