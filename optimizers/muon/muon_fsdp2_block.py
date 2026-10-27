import torch
from collections import defaultdict

from optimizers.muon.muon_utils import muon_scaling, zeropower_via_newtonschulz5_bmm, newton_schulz_factory


class MuonFSDP2Block(torch.optim.Optimizer):
    def __init__(self, params, lr, num_blocks, coeffs_type, scaling_type, weight_decay, momentum):
        # num_blocks is added only for interface compatibility
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, coeffs_type=coeffs_type, scaling_type=scaling_type))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            mu = group["momentum"]
            scaling_type = group["scaling_type"]

            buckets = defaultdict(lambda: {"params": [], "updates": []})
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["M"] = torch.zeros_like(p)

                M = state["M"]
                G_local = p.grad.to_local()
                M_local = M.to_local()

                M_local.lerp_(G_local, 1 - mu)

                N_local = G_local.lerp_(M_local, mu) # nesterov=True implicitly

                buckets[N_local.shape]["params"].append(p)
                buckets[N_local.shape]["updates"].append(N_local)

            for bucket in buckets.values():
                param_list = bucket["params"]
                Z_stack = torch.stack(bucket["updates"])

                # U_stack = zeropower_via_newtonschulz5_bmm(Z_stack, steps=5)
                U_stack = newton_schulz_factory(
                    X=Z_stack,
                    steps=5,
                    coeffs_type=group["coeffs_type"],
                    is_batched=True
                )
                for i, p in enumerate(param_list):
                    scaling = muon_scaling(p, scaling_type)
                    p.mul_(1 - lr * wd)
                    p.to_local().add_(U_stack[i], alpha=-lr * scaling)
