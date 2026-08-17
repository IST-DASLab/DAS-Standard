import torch
from torch.distributed.tensor import distribute_tensor

from optimizers.muon.muon_utils import zeropower_via_newtonschulz5


class MuonFSDP2Redundant(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, momentum):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["M"] = torch.zeros_like(p) # momentum buffer, sharded

                lr = group["lr"]
                wd = group["weight_decay"]
                mu = group["momentum"]

                G = p.grad
                M = state["M"]
                M.lerp_(G, 1 - mu) # element-wise operation, sharded

                Z = G.lerp_(M, mu).full_tensor() # full_tensor() triggers all-gather across ranks

                # at this point, all ranks have exactly the same Z matrix

                U = zeropower_via_newtonschulz5(Z, steps=5) # this is the redundancy: all ranks perform exactly the same computation (simple solution)

                coef = max(1, U.size(-2) / U.size(-1)) ** 0.5 # scaling constant
                # U.mul_(coef) # intentionally skip this operation and pack it into the learning rate

                p.mul_(1 - lr * wd)
                p.add_(
                    distribute_tensor(
                        tensor=U,
                        device_mesh=p.device_mesh,
                        placements=p.placements,
                        src_data_rank=None, # this means "no broadcasting, just take my local slice",
                    ),
                    alpha=-lr * coef
                )