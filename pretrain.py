import math
import torch
import torch.distributed as dist

from backends import get_backend, BaseBackend
from configs import get_dynamic_arg_parser
from data import get_data_readers
from evaluation import eval_and_log
from models import get_model
from optimizers import MultiOptimizer
from managers import TimingManager, LoggingManager
from constants import (
    TEXT_TRAIN_LOSS,
    TEXT_TRAIN_PPL,
    TEXT_TRAIN_GRAD_NORM,

    TEXT_ELAPSED_OPTIM_STEP,
    TEXT_ELAPSED_FORWARD,
    TEXT_ELAPSED_BACKWARD,
)


def main():
    cfg = get_dynamic_arg_parser()

    logger = LoggingManager(cfg)
    timer = TimingManager()
    backend: BaseBackend = get_backend(cfg)

    model = backend.wrap_model(get_model(cfg))
    logger.log_model_size(model)

    logger.log_console("Creating optimizer")
    optimizer = MultiOptimizer(cfg, model)

    logger.log_console("Creating data readers")
    train_reader, val_reader = get_data_readers(cfg)

    logger.start_train_pbar(total_iters=cfg.training.max_iters)

    for crt_iter in range(1, 1 + cfg.training.max_iters):
        #####
        ##### Eval
        #####
        eval_and_log(cfg, backend, model, train_reader, logger, crt_iter)

        #####
        ##### Training
        #####
        model.train()
        total_step_loss = 0.0
        for acc_step in range(cfg.training.acc_steps):
            # all-reduce gradients only at the last grad acc step
            model.require_backward_grad_sync = (acc_step == cfg.training.acc_steps - 1)
            x, y = train_reader.sample_batch()

            #####
            ##### Forward pass
            #####
            with timer.record(TEXT_ELAPSED_FORWARD):
                with backend.get_amp_context():
                    unscaled_loss = model(x, y)

            scaled_loss = unscaled_loss / cfg.training.acc_steps

            #####
            ##### Backward pass
            #####
            with timer.record(TEXT_ELAPSED_BACKWARD):
                with backend.get_amp_context():
                    scaled_loss.backward()

            total_step_loss += scaled_loss.item()
        # end for acc_step

        #####
        ##### Gradient clipping
        #####
        if cfg.training.grad_clipping > 0: # clip gradient if needed
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clipping).item()
        else: # otherwise, compute the global gradient norm
            grad_norm = math.sqrt(sum(p.grad.norm(p=2) ** 2 for p in model.parameters() if p.grad is not None))

        #####
        ##### Optimizer step
        #####
        with timer.record(TEXT_ELAPSED_OPTIM_STEP):
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        #####
        ##### Logging
        #####
        logger.update_train_pbar(step_inc=1, loss=f"{total_step_loss:.4f}")

        logger.log_dict({
            TEXT_TRAIN_LOSS: total_step_loss,
            TEXT_TRAIN_PPL: math.exp(total_step_loss),
            TEXT_TRAIN_GRAD_NORM: grad_norm,
        })
        logger.log_dict(timer.get_then_reset())
        logger.log_memory_usage()
        logger.log_redundant_iters(crt_iter) # log crt_iter for each section
        logger.commit(step=crt_iter)
    # end for crt_iter

    logger.finish()
    backend.terminate()

if __name__ == '__main__':
    main()
