import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor

from optimizers.muon.muon_utils import zeropower_via_newtonschulz5, shard_size_dim0, muon_scaling, newton_schulz_factory


class MuonFSDP2All2All(torch.optim.Optimizer):
    def __init__(self, params, lr, coeffs_type, scaling_type, weight_decay, momentum):
        params = sorted(params, key=lambda p: p.numel(), reverse=True) # sort to have balanced groups
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, coeffs_type=coeffs_type, scaling_type=scaling_type))

    @torch.no_grad()
    def step(self, closure=None):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = torch.cuda.current_device()

        for group in self.param_groups:
            params = group["params"]

            lr = group["lr"]
            wd = group["weight_decay"]
            mu = group["momentum"]

            for base in range(0, len(params), world_size): # inc base by WS
                chunk = params[base : base + world_size] # select a chunk of `world_size` params (or less if len(params) is not divisible by world_size)

                # sharded momentum update: send[i] stores the shard of chunk[i]'s update
                send = [torch.empty(0, device=device) for _ in range(world_size)] # create WS placeholders
                for i, p in enumerate(chunk):
                    state = self.state[p]
                    if len(state) == 0:
                        state["M"] = torch.zeros_like(p)

                    G = p.grad
                    M = state["M"]
                    M.lerp_(G, 1 - mu)
                    N = G.lerp_(M, mu)

                    send[i] = N.to_local().contiguous() # save shard of nesterov momentum in send[i] for this rank

                # step 1: all-to-all #1: receive every rank's shard
                if rank < len(chunk):
                    p_rank = chunk[rank]
                    sizes = shard_size_dim0(p_rank.shape[0], world_size)
                    recv = [
                        torch.empty(s, *p_rank.shape[1:], device=device, dtype=send[rank].dtype)
                        for s in sizes
                    ]
                else:
                    recv = [torch.empty(0, device=device) for _ in range(world_size)]

                dist.all_to_all(output_tensor_list=recv, input_tensor_list=send) # idle ranks will still participate, even though they have zero size

                # step 2: Newton-Schulz computation - at this point, each worker has the full nesterov matrix `Z` to orthogonalize
                if rank < len(chunk):
                    Z = torch.cat(recv, dim=0)
                    U = newton_schulz_factory(
                        X=Z,
                        steps=5,
                        coeffs_type=group["coeffs_type"],
                        is_batched=False
                    )
                    U *= muon_scaling(U, group["scaling_type"])
                    back_send = list(U.to(dtype=send[rank].dtype).split(sizes, dim=0)) # split the result according to the sizes
                else:
                    back_send = [torch.empty(0, device=device) for _ in range(world_size)]

                # step 3: all-to-all #2 - send back the results
                back_recv = [torch.empty_like(s) for s in send]
                dist.all_to_all(output_tensor_list=back_recv, input_tensor_list=back_send)

                for p, u in zip(chunk, back_recv):
                    p.mul_(1 - lr * wd)
                    p.to_local().add_(u, alpha=-lr)