import torch
from optimizers.muon.muon_utils import muon_scaling, block_muon_update

class MuonFSDP2Block(torch.optim.Optimizer):
    def __init__(self, params, lr, scaling_type, block_size, weight_decay, momentum):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, scaling_type=scaling_type, block_size=block_size))

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
                # block_size = group["block_size"] # added for compatibility in __init__; we send num rows of shard in block_muon_update

                G = p.grad
                M = state["M"]
                localR, localC = G.shape

                localU = block_muon_update(grad=G,
                                           momentum=M,
                                           block_size=localR,
                                           beta=mu,
                                           ns_steps=5,
                                           nesterov=True)

                scaling = muon_scaling(p, group["scaling_type"]) # computed using the shapes of the global tensor
                p.mul_(1 - lr * wd)
                p.add_(localU, alpha=-lr * scaling)