import torch
import torch.distributed as dist
from optimizers.muon.muon_utils import muon_scaling, zeropower_via_newtonschulz5_bmm


class MuonSingleBlock(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """

    def __init__(self, params, lr=0.02, scaling_type='kj', weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, scaling_type=scaling_type)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                lr = group["lr"]
                wd = group["weight_decay"]
                mu = group["momentum"]
                scaling = muon_scaling(p, group["scaling_type"])

                G = p.grad
                M = state["momentum_buffer"]
                WS = dist.get_world_size()

                R, C = G.shape

                M.lerp_(G, 1 - mu)
                N = G.lerp(M, mu)  # force nesterov

                assert R % WS == 0, f"Row dimension must be divisible by world size!"

                N_3d = N.view(WS, R // WS, C)
                U_3d = zeropower_via_newtonschulz5_bmm(N_3d, steps=5)
                U_2d = U_3d.view(R, C)

                p.mul_(1 - lr * wd)
                p.add_(U_2d.reshape(p.shape), alpha=-lr * scaling)

        return loss