"""Signal S1 (spec Section 2): pixelwise variance across K degradation-
consistent stochastic samples, channel-averaged. High variance = large
null-space = hallucination risk inherent to the ill-posedness itself.
"""
import torch


def compute_null_space_variance(samples: list[torch.Tensor]) -> torch.Tensor:
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples to compute variance")
    stacked = torch.stack(samples, dim=0)  # (K, C, H, W)
    var = stacked.var(dim=0, unbiased=True)  # (C, H, W)
    return var.mean(dim=0)  # (H, W)
