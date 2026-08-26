"""Range-null space projection for the average-pool downsampling operator A.

For A = average-pool by `scale`, the pseudo-inverse correction has a closed
form: within every scale x scale block, replace the block by
(block - block_mean + y_pixel). This exactly enforces A(x_proj) == y (the
range-space is fixed to the observation) while leaving the null-space
(within-block detail, i.e. anything A maps to zero) untouched. This is the
DDNM-style projection referenced in spec Section 2, Signal 1.
"""
import torch


def ddnm_project(x_hat: torch.Tensor, y: torch.Tensor, scale: int) -> torch.Tensor:
    """x_hat: (C, H, W). y: (C, H//scale, W//scale). Returns (C, H, W)."""
    c, h, w = x_hat.shape
    blocks = x_hat.reshape(c, h // scale, scale, w // scale, scale)
    block_mean = blocks.mean(dim=(2, 4), keepdim=True)
    y_expanded = y.reshape(c, h // scale, 1, w // scale, 1)
    x_proj = blocks - block_mean + y_expanded
    return x_proj.reshape(c, h, w)
