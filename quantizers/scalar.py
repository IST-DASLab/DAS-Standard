from enum import IntEnum

import torch

from .base import BaseQuantizer


class GridType(IntEnum):
    binary = 1
    ternary = 2
    int2 = 3
    int3 = 4
    int4 = 5
    int8 = 6
    fp4 = 7


def get_grid_and_bitwidth(grid_type: GridType) -> [torch.Tensor, float]:
    match grid_type:
        case GridType.binary:
            return 0.7978845587140913 * torch.linspace(-1, 1, 2, device="cuda", dtype=torch.bfloat16), 1
        case GridType.ternary:
            return 1.2240089519030855 * torch.linspace(-1, 1, 3, device="cuda", dtype=torch.bfloat16), None
        case GridType.int2:
            return 1.2240089519030855 * torch.linspace(-1, 1, 2**2, device="cuda", dtype=torch.bfloat16), 2
        case GridType.int3:
            return 2.0510679063024964 * torch.linspace(-1, 1, 2**3, device="cuda", dtype=torch.bfloat16), 3
        case GridType.int4:
            return 2.5139324513630887 * torch.linspace(-1, 1, 2**4, device="cuda", dtype=torch.bfloat16), 4
        case GridType.int8:
            return 3.884997364699907 * torch.linspace(-1, 1, 2**8, device="cuda", dtype=torch.bfloat16), 8
        case GridType.fp4:
            return 2.92247856 / 6.0 * torch.tensor(
                [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                device="cuda", dtype=torch.bfloat16,
            ), 4
        case _:
            raise ValueError(f"GridType {grid_type} unsupported")

def get_absmax_grid_and_bitwidth(grid_type: GridType) -> [torch.Tensor, float]:
    match grid_type:
        case GridType.fp4:
            return torch.tensor(
                [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                device="cuda", dtype=torch.bfloat16,
            ), 4
        case _:
            raise ValueError(f"GridType {grid_type} not yet supported for absmax")

@torch.compile
def rtn(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    inds = torch.bucketize(x, grid)

    lo = torch.clamp(inds - 1, min=0, max=grid.size(0) - 1)
    hi = torch.clamp(inds,     min=0, max=grid.size(0) - 1)

    g_lo = grid[lo]
    g_hi = grid[hi]

    pick_hi = (g_hi - x) <= (x - g_lo)
    return torch.where(pick_hi, g_hi, g_lo)


class ScQuestQ(BaseQuantizer):
    def __init__(self, grid_type="fp4", group_size=128, clip_factor=1.0, masking="l1", debias=False):
        if isinstance(grid_type, str):
            grid_type = GridType[grid_type]
            
        grid, bitwidth = get_grid_and_bitwidth(grid_type)
        
        super().__init__(bitwidth)
        self.register_buffer(
            "grid", grid,
            persistent=False,
        )
        self.group_size = group_size
        self.clip_factor = clip_factor
        self.masking = masking
        self.debias = debias

    @torch.compiler.disable()
    def forward(self, x):
        orig_shape = x.shape
        
        x = x.reshape(-1, self.group_size)
        
        with torch.no_grad():
            std = torch.std(x, correction=0, dim=-1, keepdim=True)
            x_normalized = x / std
            x_q = rtn(x_normalized, self.grid)
            
            if self.debias:
                num = (x_normalized * x_normalized).sum(dim=-1, keepdim=True)
                denom = (x_normalized * x_q).sum(dim=-1, keepdim=True)
                scales_fix = num / denom
                scales_fix = torch.where(scales_fix.isnan(), 1.0, scales_fix)
                x_q = x_q * scales_fix * std
            else:
                x_q = x_q * std
            
        if self.masking == "l1":
            pass_mask = torch.logical_and(
                x_normalized >= self.grid.min() * self.clip_factor,
                x_normalized <= self.grid.max() * self.clip_factor,
            )
            x_pass = x * pass_mask.to(x.dtype)
        elif self.masking == "none":
            x_pass = x
        else:
            raise ValueError(f"Incorrect masking scheme {self.masking}")

        return (x_pass + (x_q - x_pass).detach()).reshape(*orig_shape)

class ScAbsMaxQ(BaseQuantizer):
    def __init__(self, grid_type="fp4", group_size=128):
        if isinstance(grid_type, str):
            grid_type = GridType[grid_type]

        grid, bitwidth = get_absmax_grid_and_bitwidth(grid_type)

        super().__init__(bitwidth)
        self.register_buffer("grid", grid, persistent=False)
        self.group_size = group_size
        self.grid_max = grid.max()

    def forward(self, x):
        orig_shape = x.shape

        x = x.reshape(-1, self.group_size)

        with torch.no_grad():
            absmax = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
            scale = absmax / self.grid_max
            x_normalized = x / scale
            x_q = rtn(x_normalized, self.grid) * scale

        return (x + (x_q - x).detach()).reshape(*orig_shape)