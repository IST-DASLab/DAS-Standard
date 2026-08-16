import pathlib

import torch

from .base import BaseQuantizer


GRIDS = {
    bits: torch.load(f"{pathlib.Path(__file__).parent.resolve()}/MSE-8D-{bits:.2f}bit.pt", map_location="cpu")
    for bits in [1.00]
}

GRID_NORMS = {bits: torch.linalg.norm(GRIDS[bits], dim=1) ** 2 for bits in GRIDS.keys()}


@torch.compile
def round_to_grid(x: torch.Tensor, grid: torch.Tensor, grid_norm: torch.Tensor) -> torch.Tensor:
    idx = torch.argmax(2 * x.view(-1, grid.size(-1)) @ grid.T - grid_norm, dim=-1)
    return grid[idx].reshape_as(x)


class HiggsQuestQuantizer(BaseQuantizer):
    def __init__(self, bits, group_size=128, clip_factor=1.0, masking="l1", debias=False):        
        super().__init__(bits)
        self.register_buffer(
            "grid", GRIDS[bits].cuda(),
            persistent=False,
        )
        self.register_buffer(
            "grid_norm", GRID_NORMS[bits].cuda(),
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
            x_q = round_to_grid(x_normalized, self.grid, self.grid_norm)
            
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
