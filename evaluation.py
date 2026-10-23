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

@torch.no_grad()
def eval_and_log(cfg, backend, model, reader, logger, crt_iter):
    is_random_model = crt_iter == 0 # called with explicit crt_iter=0 before the training loop on the randomly initialized model
    is_middle_step = crt_iter % cfg.training.eval_interval == 0
    is_last_step = crt_iter == cfg.training.max_iters

    if is_random_model or is_middle_step or is_last_step:
        model.eval()
        num_batches = reader.num_batches if is_last_step else cfg.training.eval_batches_per_worker
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=backend.get_device())

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

        if backend.world_size > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        total_loss = total_loss.item() / backend.world_size / num_batches
        total_ppl = math.exp(total_loss)

        text_loss = TEXT_FINAL_VAL_LOSS if is_last_step else TEXT_VAL_LOSS
        text_ppl = TEXT_FINAL_VAL_PPL if is_last_step else TEXT_VAL_PPL

        logger.log_dict({text_loss: total_loss, text_ppl: total_ppl})
        logger.log_console(
            f"\nEvaluation",
            data=OrderedDict(
                step=crt_iter,
                num_batches=num_batches,
                loss=f"{total_loss:.4f}",
                ppl=f"{total_ppl:.4f}"# if total_ppl < 1000 else "too large (>1000)"
            )
        )
