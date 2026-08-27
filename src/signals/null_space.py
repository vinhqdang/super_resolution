"""Signal S1 (spec Section 2): pixelwise variance across K degradation-
consistent stochastic samples, channel-averaged. High variance = large
null-space = hallucination risk inherent to the ill-posedness itself.
"""
import torch

# Theoretical max of the unbiased 2-sample variance ((x1-x2)^2/2) for pixel
# values bounded in [0, 1]: maximized at x1=0, x2=1 (or vice versa), giving
# 0.5. Used as a normalization scale for K=2 (the production default in
# src/fusion/train.py's build_signal_stack); for K>2 this is a looser bound
# (the true K-sample max is lower), so normalized values stay safely <= 1
# via the explicit clamp rather than becoming a precise per-K calibration.
# Real diffusion/DDNM outputs aren't strictly guaranteed to stay in [0, 1]
# (observed slightly out-of-range values in practice), so this is a
# reasonable domain-motivated scale, not an exact mathematical guarantee —
# same caveat as S3's _MAX_EXPECTED_DISTANCE calibration.
_MAX_EXPECTED_VARIANCE_K2 = 0.5


def compute_null_space_variance(samples: list[torch.Tensor]) -> torch.Tensor:
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples to compute variance")
    stacked = torch.stack(samples, dim=0)  # (K, C, H, W)
    var = stacked.var(dim=0, unbiased=True)  # (C, H, W)
    return var.mean(dim=0)  # (H, W)


def normalized_null_space_variance(samples: list[torch.Tensor]) -> torch.Tensor:
    """[0, 1]-clamped version of compute_null_space_variance, for feeding
    into the fusion head alongside S2/S3/S4 (all already bounded to
    [0, 1]-ish ranges) — see FusionHead's other input channels in
    src/fusion/train.py's build_signal_stack. Unlike FeatureBank's
    self-calibrated normalized_distance (src/signals/distribution_shift.py),
    this uses a fixed domain-derived scale rather than a per-call
    self-calibration, since null-space variance doesn't have an analogous
    "reference set" to calibrate against."""
    raw = compute_null_space_variance(samples)
    return torch.clamp(raw / _MAX_EXPECTED_VARIANCE_K2, max=1.0)
