import math
import torch
from backends import get_backend
from configs import get_dynamic_arg_parser
from data import get_data_readers
from models import get_model
from optimizers import MultiOptimizer
from managers import TimingManager, LoggingManager


def main():
    cfg = get_dynamic_arg_parser()

    logger = LoggingManager(cfg)
    timer = TimingManager()
    backend = get_backend(cfg)

    model = backend.wrap_model(get_model(cfg))
    optimizer = MultiOptimizer(cfg, model)
    train_reader, val_reader = get_data_readers(cfg)

    for crt_iter in range(cfg.training.max_iters):
        total_step_loss = 0.0
        for acc_step in range(cfg.training.acc_steps):
            model.require_backward_grad_sync = (acc_step == cfg.training.acc_steps - 1) # all-reduce gradients only at the last grad acc step

            #####
            ##### Sample a batch of data from the training set
            #####
            x, y = train_reader.sample_batch()

            #####
            ##### Forward pass
            #####
            with timer.record("forward"):
                with backend.get_amp_context():
                    unscaled_loss = model(x, y)

            scaled_loss = unscaled_loss / cfg.training.acc_steps

            #####
            ##### Backward pass
            #####
            with timer.record("backward"):
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
        with timer.record("optimizer_step"):
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        #####
        ##### Eval
        #####


        #####
        ##### Logging
        #####
        logger.log_dict({
            'iter': crt_iter,

            'train/loss': total_step_loss,
            'train/ppl': math.exp(total_step_loss),
            'train/grad_norm': grad_norm,
        })
        logger.log_dict(timer.get_then_reset())
        logger.log_memory_usage()
        logger.commit(step=crt_iter)




    # end for iter

    logger.finish()
    backend.terminate()

if __name__ == '__main__':
    main()
