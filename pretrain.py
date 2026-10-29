import os

# Suppress C++ backend warnings and Dynamo internal logs
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_LOGS"] = "-all"

import warnings
# warnings.filterwarnings("ignore")

import math
from collections import OrderedDict

from backends import get_backend, BaseBackend
from config import get_dynamic_arg_parser
from datasets import get_data_readers
from managers import LoggingManager, TimingManager, CheckpointManager
from models import get_model
from optimizers import MultiOptimizer
from qat import wrap_model_for_qat
from utils import * # imports eval_and_log and all TEXT_* variables

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

    #####
    ##### Checkpointer
    #####
    chkpt_mgr = CheckpointManager(cfg)
    start_iter = chkpt_mgr.setup_resume(model, optimizer, train_reader)

    #####
    ##### Eval for the randomly initialized model or the loaded checkpoint
    #####
    eval_and_log(cfg, backend, model, val_reader, logger, crt_iter=0 if start_iter == 1 else start_iter-1)

    #####
    ##### Training Loop
    #####
    logger.start_train_pbar(start_iter=start_iter - 1, total_iters=cfg.training.max_iters)
    for crt_iter in range(start_iter, 1 + cfg.training.max_iters):
        model.train()
        total_step_loss = 0.0

        for acc_step in range(cfg.training.grad_acc_steps): # gradient accumulation loop
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
        ##### Checkpointing
        #####
        if (crt_iter % cfg.training.chkpt_freq == 0) or (crt_iter == cfg.training.max_iters):
            chkpt_mgr.save_checkpoint(crt_iter, model, optimizer, train_reader)

        #####
        ##### Logging
        #####
        logger.log_all_stats(
            crt_iter=crt_iter,
            total_step_loss=total_step_loss,
            grad_norm=grad_norm,
            timer=timer,
            optimizer=optimizer,
        )
    # end for crt_iter

    logger.finish()
    backend.terminate()

if __name__ == '__main__':
    main()
