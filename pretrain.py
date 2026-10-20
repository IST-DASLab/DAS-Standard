import os

# Suppress C++ backend warnings and Dynamo internal logs
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_LOGS"] = "-all"

import warnings
warnings.filterwarnings("ignore")

import math
from collections import OrderedDict

from backends import get_backend, BaseBackend
from checkpointer import Checkpointer
from config import get_dynamic_arg_parser
from constants import *
from data import get_data_readers
from evaluation import eval_and_log
from models import get_model
from managers import LoggingManager, TimingManager
from optimizers import MultiOptimizer
from qat import wrap_model_for_qat

def main():
    cfg = get_dynamic_arg_parser()
    backend: BaseBackend = get_backend(cfg) # must be created first

    logger = LoggingManager(cfg)
    timer = TimingManager()

    raw_model = get_model(cfg)
    wrap_model_for_qat(raw_model, cfg, except_modules=None)
    logger.log_config_and_model(raw_model, backend.world_size)  # use raw model before wrapping

    model = backend.wrap_model_then_compile(raw_model) # wrap the model first, so the params are now DTensors
    optimizer = MultiOptimizer(cfg, model, backend) # build optimizer: params are now DTensors

    train_reader, val_reader = get_data_readers(cfg)

    logger.log_console(f"Dataset tokens", data=OrderedDict(train=f"{train_reader.num_toks:_}", val=f"{val_reader.num_toks:_}"))
    logger.start_train_pbar(total_iters=cfg.training.max_iters)

    #####
    ##### Eval for the randomly initialized model
    #####
    eval_and_log(cfg, backend, model, val_reader, logger, crt_iter=0)

    #####
    ##### Training Loop
    #####
    for crt_iter in range(1, 1 + cfg.training.max_iters):
        #####
        ##### Training
        #####
        model.train()
        total_step_loss = 0.0
        for acc_step in range(cfg.training.grad_acc_steps):
            # all-reduce gradients only at the last grad acc step
            should_sync_grad = (acc_step == cfg.training.grad_acc_steps - 1)
            backend.set_grad_sync(model, enabled=should_sync_grad)
            x, y = train_reader.sample_batch()

            #####
            ##### Forward pass
            #####
            with timer.record(TEXT_ELAPSED_FORWARD):
                with backend.get_amp_context():
                    unscaled_loss = model(x, y)

            scaled_loss = unscaled_loss / cfg.training.grad_acc_steps

            #####
            ##### Backward pass
            #####
            # TODO: use grad scaler for fp16
            with timer.record(TEXT_ELAPSED_BACKWARD):
                with backend.get_amp_context():
                    scaled_loss.backward()

            total_step_loss += scaled_loss.item()
        # end for acc_step

        #####
        ##### Gradient clipping
        #####
        grad_norm = backend.clip_grad_norm_(model, max_norm=cfg.optimizer.grad_clipping)

        #####
        ##### Optimizer step
        #####
        with timer.record(TEXT_ELAPSED_OPTIM_STEP):
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        #####
        ##### Eval
        #####
        with timer.record(TEXT_ELAPSED_EVAL):
            eval_and_log(cfg, backend, model, val_reader, logger, crt_iter)

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
        logger.log_dict(optimizer.extract_learning_rates())
        logger.log_dict(optimizer.get_optimzier_stats())
        logger.log_memory_usage()
        logger.log_xaxis(
            iter=crt_iter,
            tokens=crt_iter * cfg.training.local_batch_size * cfg.training.grad_acc_steps * cfg.model.seq_len,
        ) # log these values to all panel grids (section) in wandb
        logger.commit(step=crt_iter)
    # end for crt_iter

    logger.finish()
    backend.terminate()

if __name__ == '__main__':
    main()
