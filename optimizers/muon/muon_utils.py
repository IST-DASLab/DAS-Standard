from typing import List

import torch
import math
import torch.distributed as dist
from torch.distributed.tensor import DTensor

def newton_schulz_factory(X, steps, coeffs_type, is_batched):
    is_cubic = "p3" in coeffs_type
    is_quintic = "p5" in coeffs_type

    if is_cubic:
        ns_fn = zeropower_via_newtonschulz3_bmm if is_batched else zeropower_via_newtonschulz3
    elif is_quintic:
        ns_fn = zeropower_via_newtonschulz5_bmm if is_batched else zeropower_via_newtonschulz5
    else:
        raise RuntimeError(f"Unknown coeffs type for Muon: {coeffs_type}")

    return ns_fn(X, steps, coeffs_type)

def zeropower_via_newtonschulz5(G, steps: int, coeffs_type: str):
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    a_b_c_coeffs: List[tuple] = muon_coeffs(coeffs_type)
    for i in range(steps):
        a, b, c = a_b_c_coeffs[i]
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def zeropower_via_newtonschulz5_bmm(G, steps: int, coeffs_type: str):
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    a_b_c_coeffs: List[tuple] = muon_coeffs(coeffs_type)
    for i in range(steps): # X has shape (N, R / N, C / N) => N * (R / N) * (C / N) = R * C / N
        a, b, c = a_b_c_coeffs[i]
        A = torch.bmm(X, X.mT)
        B = b * A + c * torch.bmm(A, A)  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + torch.bmm(B, X)

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def zeropower_via_newtonschulz3(G, steps: int, coeffs_type: str):
    assert G.ndim >= 2
    X = G.bfloat16()

    R, C = G.shape
    if R > C:
        X = X.mT

    idx = torch.arange(min(R, C)).to(G.device)

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    a_b_coeffs: List[tuple] = muon_coeffs(coeffs_type)
    for i in range(steps):
        a, b = a_b_coeffs[i]
        XXt = X @ X.mT
        XXt.mul_(b)
        XXt[idx, idx] += a
        X = XXt @ X

    if R > C:
        X = X.mT
    return X

def zeropower_via_newtonschulz3_bmm(G, steps: int, coeffs_type: str):
    assert G.ndim >= 2
    X = G.bfloat16()

    _, R, C = G.shape
    if R > C:
        X = X.mT

    idx = torch.arange(min(R, C)).to(G.device)

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    a_b_coeffs: List[tuple] = muon_coeffs(coeffs_type)
    for i in range(steps):
        a, b = a_b_coeffs[i]
        XXt = torch.bmm(X, X.mT)
        XXt.mul_(b)
        XXt[:, idx, idx] += a
        X = torch.bmm(XXt, X)

    if R > C:
        X = X.mT
    return X


def muon_scaling(p, scaling_type):
    R, C = p.shape
    match scaling_type:
        case 'none':
            return 1
        case 'kj':
            return max(1, R / C) ** 0.5
        case 'rms':
            return 0.2 * math.sqrt(max(R, C))
        case 'block':
            return 0.2 * math.sqrt(max(R / dist.get_world_size(), C))
        case 'dion':
            return (R / C) ** 0.5
        case _:
            raise RuntimeError(f"Unknown scaling type: {scaling_type}, should be 'kj', 'none', 'rms', 'block' or 'dion'!")

def muon_coeffs(muon_coeffs_type, polar_express_safety_factor=1.01):
    match muon_coeffs_type.lower():
        case "p5-you":
            # 5th order polynomial by You https://x.com/YouJiacheng/status/1905861218138804534
            # code: https://gist.github.com/YouJiacheng/6f4de2fac9b4edd8bcb2df7058652a7f#file-train_gpt-20250329-txt-L125
            return [
                [4.0848, -6.8946, 2.9270],
                [3.9505, -6.3029, 2.6377],
                [3.7418, -5.5913, 2.3037],
                [2.8769, -3.1427, 1.2046],
                [2.8366, -3.0525, 1.2012]
            ]
        case "p5-polar":
            # polar express: https://arxiv.org/pdf/2505.16932#page=16
            raw_polar_express = [
                (8.28721201814563, -23.595886519098837, 17.300387312530933),
                (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
                (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
                (3.3184196573706015, -2.488488024314874, 0.51004894012372),
                (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
                (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
                (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
                (1.875, -1.25, 0.375), # subsequent coeffs equal this numerically
            ]
            sf = polar_express_safety_factor
            return [
                (a / sf, b / (sf ** 3), c / (sf ** 5))
                for a, b, c in raw_polar_express[:-1]
            ] + [raw_polar_express[-1]]
        case "p5-book":
            # comes from the polynomial p(x) = (15x - 10x^3 + 3x^5) / 8 (can be found in the book)
            return [
                (1.875, -1.25, 0.375)
            ] * 5
        case "p5-kj":
            # original from Keller Jordan
            return [
                (3.4445, -4.7750, 2.0315)
            ] * 5
        case "p3-nv":
            # 3rd order polynomial from Nvidia's paper https://arxiv.org/pdf/2606.00371
            return [
                (3.3656576, -3.3420992),
                (2.5744352, -1.4957376),
                (2.5368962, -1.4312570),
                (2.4418906, -1.2764040),
                (2.2230472, -0.9630650),
            ]
        case "p3-book":
            # 3rd order polynomial p(x) = 1.5x - 0.5x^3 (can be found in the book)
            return [
                (1.5, -0.5),
            ] * 5
        case _:
            raise RuntimeError(f"Unknown muon_coeffs_type: {muon_coeffs_type}")

def shard_size_dim0(num_rows: int, world_size: int):
    """
    Split `num_rows` rows into chunks of size `world_size`
    """
    chunks = torch.arange(num_rows).chunk(world_size)
    sizes = [len(c) for c in chunks]
    return sizes + [0] * (world_size - len(sizes))
