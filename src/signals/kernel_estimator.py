"""Lightweight blind degradation estimator (simplified MANet-style): a
small CNN regressing degradation parameters (sigma_x, sigma_y, theta,
noise_sigma) from a 64x64 LR patch. Used by Signal S3 (degradation
mismatch) in src/signals/degradation_mismatch.py to compare the estimated
degradation against the backbone's assumed training distribution.

Known limitations of the current MSE training objective (scripts/
train_kernel_estimator.py), left as-is for this first-pass estimator:
- The four regression targets have very different numeric scales
  (sigma in [0.2, 4.0], theta in [0, pi], noise_sigma in [0, 0.03]), so
  unweighted MSE gives noise_sigma negligible gradient signal relative to
  sigma/theta. Would need per-term normalization or loss weights to fix.
- theta is regressed as a raw angle even though the kernel is pi-periodic
  (theta and theta+pi produce the identical kernel), so values near 0 and
  near pi — physically close — are penalized as maximally different. The
  standard fix (regressing (sin theta, cos theta) instead) would change
  the model's output shape and the downstream consumer's theta_distance
  calculation in degradation_mismatch.py; deferred to keep this a
  first-pass estimator without cascading redesign.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.degradation.mismatch_degrade import degrade_mismatch
from src.degradation.real_esrgan_degrade import degrade_standard


class KernelEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),  # 64 -> 32
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 32 -> 16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 16 -> 8
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 4)

    def forward(self, lr_patch: torch.Tensor) -> torch.Tensor:
        feats = self.net(lr_patch).flatten(1)
        return self.head(feats)


class KernelEstimatorDataset(Dataset):
    """Mixes standard- and mismatch-pool degradations 50/50 so the estimator
    learns to regress kernel params across both distributions.

    Derives a fresh RNG per __getitem__ call from (seed, idx) rather than
    mutating a single shared np.random.Generator instance: with
    num_workers>0, DataLoader forks/spawns worker processes that each get
    an unmutated copy of the Dataset object, so a stateful shared RNG
    starts from the same seed in every worker and gets replayed
    identically every epoch (only which image a draw lands on changes via
    shuffle) — silently cutting effective degradation diversity. A
    per-item RNG keyed on idx is deterministic, reproducible, and
    independent of worker count or epoch. Trade-off: each item now gets
    the SAME degradation instance on every epoch (rather than a fresh
    resample), trading per-epoch diversity for correctness — true
    per-epoch resampling would need an epoch-aware seed threaded in from
    the training loop, deferred as unnecessary complexity for this
    first-pass estimator (dataset size and epoch count control diversity
    another way: use more source images rather than more resamples)."""

    def __init__(self, hr_images: list[np.ndarray], scale: int = 4, seed: int = 0):
        self.hr_images = hr_images
        self.scale = scale
        self.seed = seed

    def __len__(self) -> int:
        return len(self.hr_images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hr = self.hr_images[idx]
        rng = np.random.default_rng((self.seed, idx))
        if rng.random() < 0.5:
            lr, params = degrade_standard(hr, self.scale, rng)
        else:
            lr, params = degrade_mismatch(hr, self.scale, rng)
        lr_t = torch.from_numpy(lr).permute(2, 0, 1).float()
        target = torch.tensor([params["sigma_x"], params["sigma_y"], params["theta"], params["noise_sigma"]]).float()
        return lr_t, target
