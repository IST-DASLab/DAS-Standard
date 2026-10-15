import torch
import math


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def zeropower_via_newtonschulz5_bmm(G, steps: int):
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Perform the NS iterations
    for _ in range(steps): # X has shape (N, R / N, C / N) => N * (R / N) * (C / N) = R * C / N
        A = torch.bmm(X, X.mT)
        B = b * A + c * torch.bmm(A, A)  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + torch.bmm(B, X)

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def muon_scaling(p, scaling_type):
    R, C = p.shape
    match scaling_type:
        case 'kj':
            return max(1, R / C) ** 0.5
        case 'none':
            return 1
        case 'kimi':
            return 0.2 * math.sqrt(max(R, C))
        case 'dion':
            return (R / C) ** 0.5
        case _:
            raise RuntimeError(f"Unknown scaling type: {scaling_type}, should be 'kj', 'none', 'kimi' or 'dion'!")

def extract_all_blocks(layer, B):
    R, C = layer.shape

    # Calculate grid dimensions
    num_row_blocks = R // B
    num_col_blocks = C // B

    # 1. Split into (num_row_blocks, B, num_col_blocks, B)
    view_4d = layer.view(num_row_blocks, B, num_col_blocks, B)

    # 2. Permute to (num_row_blocks, num_col_blocks, B, B)
    permuted = view_4d.permute(0, 2, 1, 3)

    # 3. Flatten the first two dimensions to N = num_row_blocks * num_col_blocks
    # Shape becomes (N, B, B)
    blocks_3d = permuted.reshape(-1, B, B)

    return blocks_3d

def reconstruct_from_blocks(blocks_3d, R, C):
    _, B, _ = blocks_3d.shape
    num_row_blocks = R // B
    num_col_blocks = C // B

    # 1. Unflatten N back into the block grid shape (num_row_blocks, num_col_blocks, B, B)
    grid_4d = blocks_3d.view(num_row_blocks, num_col_blocks, B, B)

    # 2. Undo the permutation to interleave the block rows and columns again
    unpermuted = grid_4d.permute(0, 2, 1, 3)

    # 3. Reshape back to the original (R, C) matrix
    # Using .reshape() because .permute() makes the tensor non-contiguous in memory
    layer = unpermuted.reshape(R, C)

    return layer

def muon_ddp_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    return update

def block_muon_update(grad, momentum, block_size, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    R, C = update.shape
    if isinstance(update, torch.DTensor):
        update = update.to_local()

        R, C = update.shape # for DTensor, R will be the sharded dimension (num rows on the current worker)
        assert C % R == 0, f"For FSDP2, the local tensors should have C % R == 0, but got {C=} and {R=}"
        block_size = min(R, C)

    blocks_3d = extract_all_blocks(update, B=block_size)
    ortho_blocks_3d = zeropower_via_newtonschulz5_bmm(blocks_3d, steps=ns_steps)
    update = reconstruct_from_blocks(ortho_blocks_3d, R, C)
    del blocks_3d, ortho_blocks_3d
    return update

def shard_size_dim0(num_rows: int, world_size: int):
    """
    Split `num_rows` rows into chunks of size `world_size`
    """
    chunks = torch.arange(num_rows).chunk(world_size)
    sizes = [len(c) for c in chunks]
    return sizes + [0] * (world_size - len(sizes))