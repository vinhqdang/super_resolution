"""Signal S3 (spec Section 2): re-estimate the LR patch's degradation
kernel with the blind estimator, then score how far the estimate falls
outside the backbone's assumed (standard-pool) training distribution.
Normalized to [0, 1] via a fixed max-distance scale so it composes cleanly
with the other three signals in the fusion head.

Limitation: sigma_distance only measures how far max(sigma_x, sigma_y)
exceeds the standard pool's upper bound. It cannot detect pure anisotropy
where both sigmas individually stay within STANDARD_SIGMA_RANGE (the
standard pool is always isotropic, sigma_x == sigma_y, so any anisotropy
is itself a departure — but this metric would miss it). This is currently
masked by the synthetic benchmark's design: MISMATCH_SIGMA_RANGE is
disjoint from and strictly above STANDARD_SIGMA_RANGE, so every genuine
mismatch-pool sample already has both sigmas past the upper bound. Would
need revisiting if the mismatch pool or a real-world degradation ever
produces anisotropy without both components exceeding the standard range.

Limitation: noise_sigma is currently unused in this score. Both
degrade_standard and degrade_mismatch sample noise from the same
STANDARD_NOISE_RANGE (real_esrgan_degrade.py / mismatch_degrade.py), so
noise is never actually out-of-distribution between the two pools today —
omitting it loses no real detection power against the current benchmark.
Would need adding if a future degradation pool introduces an OOD noise
range.
"""
import torch

from src.degradation.mismatch_degrade import MISMATCH_SIGMA_RANGE, MISMATCH_THETA_RANGE
from src.degradation.real_esrgan_degrade import STANDARD_SIGMA_RANGE

# Worst-case raw distance achievable within the mismatch pool's own ranges:
# max sigma distance past the standard upper bound, plus max theta distance
# (theta is always 0 in the standard pool). Not a fixed magic number, so it
# stays correct if either pool's ranges change.
_MAX_EXPECTED_DISTANCE = (MISMATCH_SIGMA_RANGE[1] - STANDARD_SIGMA_RANGE[1]) + MISMATCH_THETA_RANGE[1]


def compute_degradation_mismatch(estimator, lr_patch: torch.Tensor) -> float:
    # Match lr_patch to the estimator's device when it's a real nn.Module
    # (KernelEstimator) — callers (build_signal_stack, run_chasr) pass the
    # caller's lr_patch as-is, which is CPU when it came straight from a
    # Dataset, while the estimator is typically moved to CUDA. Test doubles
    # (e.g. _FixedEstimator) have no .parameters(), so this is a no-op for
    # them rather than an error.
    if hasattr(estimator, "parameters"):
        try:
            lr_patch = lr_patch.to(device=next(estimator.parameters()).device)
        except StopIteration:
            pass
    out = estimator(lr_patch)
    sigma_x, sigma_y, theta, _noise_sigma = out[0].tolist()  # noise_sigma unused, see module docstring
    sigma_distance = max(0.0, max(sigma_x, sigma_y) - STANDARD_SIGMA_RANGE[1])
    theta_distance = abs(theta)  # standard pool always has theta == 0
    raw = sigma_distance + theta_distance
    return float(min(1.0, raw / _MAX_EXPECTED_DISTANCE))
