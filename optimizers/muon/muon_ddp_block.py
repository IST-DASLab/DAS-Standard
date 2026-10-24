import torch
import torch.distributed as dist
from optimizers.muon.muon_utils import muon_scaling, zeropower_via_newtonschulz5_bmm

class MuonDDPBlock(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, num_blocks=0, scaling_type='kj', weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, num_blocks=num_blocks, scaling_type=scaling_type)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization

                    state = self.state[p]

                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    lr = group["lr"]
                    wd = group["weight_decay"]
                    mu = group["momentum"]
                    num_blocks = group["num_blocks"]
                    scaling = muon_scaling(p, group["scaling_type"])

                    G = p.grad
                    M = state["momentum_buffer"]
                    B = dist.get_world_size() if num_blocks == 0 else num_blocks

                    R, C = G.shape

                    M.lerp_(G, 1 - mu)
                    N = G.lerp(M, mu) # force nesterov

                    assert R % B == 0, f"Row dimension must be divisible by world size!"

                    N_3d = N.view(B, R // B, C)
                    U_3d = zeropower_via_newtonschulz5_bmm(N_3d, steps=5)
                    U_2d = U_3d.view(R, C)

                    p.mul_(1 - lr * wd)
                    p.add_(U_2d.reshape(p.shape), alpha=-lr * scaling)
                # end if
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            # end for
        # end for
        return loss
