import math
from argparse import Namespace as ArgParseNamespace

def get_lr_cos_warmup(cfg: ArgParseNamespace, lr_init: float, step: int):
    max_iters = cfg.training.max_iters
    warmup_steps = int(max_iters * cfg.optimizer.lr_warmup)

    if step < warmup_steps:
        lr_start = lr_init * cfg.optimizer.cos_start_lr_ratio
        progress = step / max(1, warmup_steps) # prevent div by zero if warmup_steps is 0
        lr_crt = lr_start + progress * (lr_init - lr_start)
    else:
        lr_end = lr_init * cfg.optimizer.cos_end_lr_ratio
        total_decay_steps = max_iters - warmup_steps
        crt_decay_step = step - warmup_steps
        progress = crt_decay_step / max(1, total_decay_steps)
        lr_crt = lr_end + 0.5 * (lr_init - lr_end) * (1 + math.cos(math.pi * progress))

    return lr_crt
