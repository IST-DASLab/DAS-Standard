from collections import defaultdict
from contextlib import contextmanager

import torch


class TimingManager:
    def __init__(self):
        self.metrics = defaultdict(float)

    @contextmanager
    def record(self, block_name: str):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        yield # at this point __enter__ ends and python executes the code in the "with record('block')" clause

        # the code below runs after the code in with clause ends - this part of the function is equivalent to __exit__
        end_event.record()

        torch.cuda.synchronize()
        self.metrics[block_name] += start_event.elapsed_time(end_event)

    def get_then_reset(self):
        crt_metrics = self.metrics.copy()
        self.metrics.clear()
        return crt_metrics
