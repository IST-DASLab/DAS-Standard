import math
from argparse import Namespace as ArgParseNamespace

# def get_lr_cos_warmup(cfg: ArgParseNamespace, lr_init: float, step: int):
#     max_iters = cfg.training.max_iters
#     warmup_steps = int(max_iters * cfg.optimizer.lr_sched_warmup)
#
#     if step < warmup_steps:
#         lr_start = lr_init * cfg.optimizer.lr_sched_start_ratio
#         progress = step / max(1, warmup_steps) # prevent div by zero if warmup_steps is 0
#         lr_crt = lr_start + progress * (lr_init - lr_start)
#     else:
#         lr_end = lr_init * cfg.optimizer.lr_sched_end_ratio
#         total_decay_steps = max_iters - warmup_steps
#         crt_decay_step = step - warmup_steps
#         progress = crt_decay_step / max(1, total_decay_steps)
#         lr_crt = lr_end + 0.5 * (lr_init - lr_end) * (1 + math.cos(math.pi * progress))
#
#     return lr_crt


def interp(start, end, alpha):
    """
    alpha = 0 => return start
    alpha = 1 => return end
    alpha in (0, 1) => return convex combination
    """
    return (1 - alpha) * start + alpha * end

class LRScheduler:
    def __init__(self, cfg: ArgParseNamespace):
        self.cfg = cfg

    def _compute_info(self, peak):
        cfg = self.cfg
        T = cfg.training.max_iters # total number of steps
        W = int(T * cfg.optimizer.lr_sched_warmup)
        start = peak * cfg.optimizer.lr_sched_start_ratio
        end = peak * cfg.optimizer.lr_sched_end_ratio
        return start, end, T, W

    def _get_linear_lr_w_linear_warmup(self, t, peak):
        start, end, T, W = self._compute_info(peak)

        if t < W:  # warmup
            return interp(start=start, end=peak, alpha=t / max(1, W))

        return interp(start=peak, end=end, alpha=(t - W) / max(1, T - W))

    def _get_cosine_lr_w_linear_warmup(self, t, peak):
        start, end, T, W = self._compute_info(peak)

        if t < W:  # warmup
            return interp(start=start, end=peak, alpha=t / max(1, W))

        a = (t - W) / max(1, T - W)
        alpha = 0.5 * (1 + math.cos(math.pi * a))
        return interp(start=peak, end=end, alpha=alpha)

    def _get_wsd_lr_w_linear_warmup(self, t, peak):
        start, end, T, W = self._compute_info(peak)
        D = int(T * self.cfg.optimizer.lr_sched_wsd_decay)

        if t < W:  # warmup
            return interp(start=start, end=peak, alpha=t / max(1, W))

        if W <= t < D:
            return peak

        return interp(start=peak, end=end, alpha=(t - D) / max(1, T - D))


    def __call__(self, t, peak):
        """
        The linear warmup and the decays (including cosine) can be written as a
        convex combination between the start and end learning rates.

        For linear parts, the coeff of convex combination is the progress alpha
        representing the percentage of step t in the warmup/decay phase.

        For cosine decay, the coeff of convex combination is the modified cosine
        of the same percentage of step t in the warmup/decay phase.

        The schedule starts from lr_start to lr_peak to lr_end. In the code, we
        skip the "lr_" prefix and stick to start, peak, end

        :param t: optimization step
        :param peak: peak learning rate
        """
        sched_type = self.cfg.optimizer.lr_sched_type

        match sched_type:
            case "linear":
                return self._get_linear_lr_w_linear_warmup(t, peak)
            case "cosine":
                return self._get_cosine_lr_w_linear_warmup(t, peak)
            case "wsd":
                return self._get_wsd_lr_w_linear_warmup(t, peak)
            case _:
                raise NotImplementedError(f"Unknown schedule type {sched_type}")