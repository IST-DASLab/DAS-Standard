import math

import torch
import torch.distributed as dist

from constants import (
    TEXT_VAL_LOSS,
    TEXT_VAL_PPL,
    TEXT_FINAL_VAL_LOSS,
    TEXT_FINAL_VAL_PPL,
)

def log_eval_loss(cfg, logger, crt_iter, loss):
    if crt_iter == cfg.training.max_iters:
        text_loss = TEXT_FINAL_VAL_LOSS
        text_ppl = TEXT_FINAL_VAL_PPL
    else:
        text_loss = TEXT_VAL_LOSS
        text_ppl = TEXT_VAL_PPL

    logger.log_dict({
        text_loss: loss,
        text_ppl: math.exp(text_loss),
    })


@torch.no_grad()
def eval_and_log(cfg, backend, model, reader, logger, crt_iter):
    is_first_step = crt_iter == 1
    is_middle_step = crt_iter % cfg.training.eval_interval == 0
    is_last_step = crt_iter == cfg.training.max_iters

    if is_first_step or is_middle_step or is_last_step:
        model.eval()
        num_batches = cfg.training.eval_batches if crt_iter < cfg.training.max_iters else reader.num_batches
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=model.device)
        reader.reset()
        for idx in range(num_batches):
            x, y = reader.sample_batch()
            with backend.get_amp_context():
                loss = model(x, y)
            total_loss += loss.item()
        # end for idx

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        total_loss = total_loss.item() / backend.world_size

        log_eval_loss(logger, crt_iter, total_loss)