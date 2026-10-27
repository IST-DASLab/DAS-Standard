import torch
import torch.distributed as dist

from .muon_utils import muon_ddp_update, muon_scaling, newton_schulz_factory


class MuonDDP(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """

    def __init__(self, params, lr=0.02, coeffs_type="p5-kj", scaling_type='kj', weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, coeffs_type=coeffs_type, scaling_type=scaling_type)
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
                    beta = group["momentum"]

                    G = p.grad
                    M = state["momentum_buffer"]

                    M.lerp_(G, 1 - beta)
                    N = G.lerp_(M, beta)  # N for Nesterov, but it has G's storage

                    U = newton_schulz_factory(
                        X=N,
                        steps=5,
                        coeffs_type=group["coeffs_type"],
                        is_batched=False
                    )
                    scaling = muon_scaling(p, group["scaling_type"])
                    p.mul_(1 - lr * wd)
                    p.add_(U.reshape(p.shape), alpha=-lr * scaling)
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss
