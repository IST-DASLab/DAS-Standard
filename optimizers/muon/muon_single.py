import torch

from .muon_utils import muon_ddp_update, muon_scaling, newton_schulz_factory


class MuonSingle(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """

    def __init__(self, params, lr=0.02, coeffs_type="p5-kj", scaling_type='kj', weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, coeffs_type=coeffs_type, scaling_type=scaling_type)
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
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)


                lr = group["lr"]
                wd = group["weight_decay"]
                beta = group["momentum"]
                G = p.grad
                M = state["momentum_buffer"]
                M.lerp_(G, 1 - beta)
                N = G.lerp_(M, beta)  # N for Nesterov, but it has G's storage

                # update = muon_ddp_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                U = newton_schulz_factory(
                    X=N,
                    steps=5,
                    coeffs_type=group["coeffs_type"],
                    is_batched=False
                )
                scaling = muon_scaling(p, group["scaling_type"])
                p.mul_(1 - lr * wd)
                p.add_(U.reshape(p.shape), alpha=-lr * scaling)

        return loss
