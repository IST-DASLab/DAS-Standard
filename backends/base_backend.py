import random
from argparse import Namespace as ArgParseNamespace

import numpy as np
import torch
from torch.distributed.tensor import DTensor


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
