import torch
import torch.distributed as dist
from optimizers.muon.muon_utils import muon_scaling, zeropower_via_newtonschulz5_bmm, newton_schulz_factory


class MuonSingleBlock(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """

    def __init__(self, params, lr=0.02, num_blocks=0, coeffs_type="p5-kj", scaling_type='kj', weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, num_blocks=num_blocks, coeffs_type=coeffs_type, scaling_type=scaling_type)
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
                num_blocks = group["num_blocks"]
                scaling = muon_scaling(p, group["scaling_type"], num_blocks)

                G = p.grad
                M = state["momentum_buffer"]
                B = 1 if num_blocks == 0 else num_blocks

                R, C = G.shape

                M.lerp_(G, 1 - mu)
                N = G.lerp_(M, mu)  # force nesterov

                assert R % B == 0, f"Row dimension must be divisible by world size!"

                N_3d = N.view(B, R // B, C)
                # U_3d = zeropower_via_newtonschulz5_bmm(N_3d, steps=5)
                U_3d = newton_schulz_factory(
                    X=N_3d,
                    steps=5,
                    coeffs_type=group["coeffs_type"],
                    is_batched=True,
                )

                p.mul_(1 - lr * wd)
                p.add_(U_3d.reshape(p.shape), alpha=-lr * scaling)

        return loss