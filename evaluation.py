import math
import torch
import torch.distributed as dist
from collections import OrderedDict

from constants import (
    TEXT_VAL_LOSS,
    TEXT_VAL_PPL,
    TEXT_FINAL_VAL_LOSS,
    TEXT_FINAL_VAL_PPL,
)

def log_eval_loss(cfg, logger, crt_iter, loss):
    is_last_step = crt_iter == cfg.training.max_iters

    text_loss = TEXT_FINAL_VAL_LOSS if is_last_step else TEXT_VAL_LOSS
    text_ppl = TEXT_FINAL_VAL_PPL if is_last_step else TEXT_VAL_PPL

    logger.log_dict({text_loss: loss, text_ppl: math.exp(loss)})
    logger.log_console(f"\nEvaluation", data=OrderedDict(step=crt_iter, loss=f"{loss:.4f}", ppl=f"{math.exp(loss):.4f}" if loss <= 10 else "too large"))


@torch.no_grad()
def eval_and_log(cfg, backend, model, reader, logger, crt_iter):
    is_first_step = crt_iter == 1
    is_middle_step = crt_iter % cfg.training.eval_interval == 0
    is_last_step = crt_iter == cfg.training.max_iters - 1

    if is_first_step or is_middle_step or is_last_step:
        model.eval()
        num_batches = cfg.training.eval_batches if crt_iter < cfg.training.max_iters else reader.num_batches
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=model.device)

        logger.start_eval_pbar(total_batches=num_batches)
        reader.reset()

        for idx in range(num_batches):
            x, y = reader.sample_batch()
            with backend.get_amp_context():
                loss = model(x, y)
            total_loss += loss.item()

            logger.update_eval_pbar(step_inc=1, val_loss=f"{total_loss:.4f}")
        # end for idx

        logger.close_eval_pbar()

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        total_loss = total_loss.item() / backend.world_size

        log_eval_loss(cfg, logger, crt_iter, total_loss)
