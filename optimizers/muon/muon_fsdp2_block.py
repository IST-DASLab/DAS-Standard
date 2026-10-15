import torch
from collections import defaultdict

from optimizers.muon.muon_utils import muon_scaling, block_muon_update, muon_ddp_update, zeropower_via_newtonschulz5_bmm


class MuonFSDP2Block(torch.optim.Optimizer):
    def __init__(self, params, lr, scaling_type, block_size, weight_decay, momentum):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, scaling_type=scaling_type, block_size=block_size))

    # @torch.no_grad()
    # def step_sequential(self, closure=None):
    #     for group in self.param_groups:
    #         for p in group["params"]:
    #             if p.grad is None:
    #                 continue
    #
    #             state = self.state[p]
    #             if len(state) == 0:
    #                 state["M"] = torch.zeros_like(p) # momentum buffer, sharded
    #
    #             lr = group["lr"]
    #             wd = group["weight_decay"]
    #             mu = group["momentum"]
    #             # block_size = group["block_size"] # added for compatibility in __init__; we send num rows of shard in block_muon_update
    #
    #             G = p.grad
    #             M = state["M"]
    #
    #             u = muon_ddp_update(grad=G.to_local(), momentum=M.to_local(), beta=mu, ns_steps=5, nesterov=True)
    #             scaling = muon_scaling(p, group["scaling_type"]) # computed using the shapes of the global tensor
    #             p.to_local().mul_(1 - lr * wd).add_(u, alpha=-lr * scaling)

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

                U_stack = zeropower_via_newtonschulz5_bmm(Z_stack, steps=5)
                for i, p in enumerate(param_list):
                    scaling = muon_scaling(p, scaling_type)
                    p.mul_(1 - lr * wd)
                    p.to_local().add_(U_stack[i], alpha=-lr * scaling)
