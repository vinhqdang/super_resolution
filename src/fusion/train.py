"""Trains the fusion head on xSR-CausalBench. Each dataset item's signal
stack requires ~5 diffusion sampling calls per k (1 hop-1 call + 4 tiled
hop-2 calls) through the backbone (Task 5) — expensive enough that
recomputing it on every training epoch would make Phase 1 training take
days on an 8GB laptop GPU. `precompute_signal_stacks` therefore builds
every (signal_stack, label_map) pair ONCE, optionally caching to disk, and
`train` iterates epochs over that cheap, cached tensor set — only the
small fusion head does repeated forward/backward passes.

Known limitations, left as-is for this first-pass Phase 1 training loop
(documented rather than silently accepted):
- S1's k=2 sample count (build_signal_stack's default) makes the
  null-space-variance estimate high-variance itself (1 degree of freedom
  for an unbiased estimator) — flagged during Task 6's review, not
  addressed here; would need averaging across more seeds to reduce.
- The cross_entropy loss is unweighted despite class imbalance in
  xSR-CausalBench: build_ill_posed_sample/build_mismatch_sample label the
  WHOLE image one class, while build_prior_reliance_sample/
  build_distribution_shift_sample label only a small region (the rest
  RELIABLE) — the two "regional" cause classes are a minority of total
  training pixels. Would need class weighting (e.g. inverse frequency) to
  address; deferred for this first-pass loop.
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.backbone.diffusion_backbone import estimate_prior_reliance_map
from src.causalbench.build_benchmark import CAUSE_LABELS
from src.causalbench.dataset import CausalBenchDataset
from src.fusion.model import FusionHead
from src.signals.degradation_mismatch import compute_degradation_mismatch
from src.signals.distribution_shift import FeatureBank, extract_dinov2_features
from src.signals.null_space import normalized_null_space_variance

# Single source of truth for the K used to build signal stacks — S1's
# normalization (normalized_null_space_variance) is calibrated for K=2,
# and the fusion head is trained on whatever K produced its training
# signal stacks. scripts/train_fusion_head.py, scripts/run_evaluation.py,
# and src/fusion/infer.py's run_chasr default all import this rather than
# hardcoding their own value, so train/eval K can't silently drift apart
# (a real bug caught in Task 14's review: run_chasr's old k=4 default
# didn't match training's k=2, a train/inference distribution mismatch on
# S1 exactly analogous to the S1/S4 normalization bugs fixed alongside it).
SIGNAL_STACK_K = 2


def build_signal_stack(lr, backbone, kernel_estimator, feature_bank, dino_encoder, k=SIGNAL_STACK_K):
    samples = backbone.sample_k_16x(lr, k=k, base_seed=0)
    # normalized (not raw): bounds S1 to [0,1] like the other three signal
    # channels — see src/signals/null_space.py docstring.
    s1 = normalized_null_space_variance(samples)  # (H, W)

    s2 = estimate_prior_reliance_map(backbone, lr, base_seed=0)  # (H, W) — see Task 5 for why this, not compute_prior_reliance

    s3_scalar = compute_degradation_mismatch(kernel_estimator, lr.unsqueeze(0))
    s3 = torch.full_like(s1, s3_scalar)

    query_feats = extract_dinov2_features(dino_encoder, samples[0].unsqueeze(0))
    # normalized_distance (not raw knn_distance): bounds S4 to [0,1] like
    # the fusion head's other input channels — see FeatureBank docstring.
    s4_scalar = feature_bank.normalized_distance(query_feats).item()
    s4 = torch.full_like(s1, s4_scalar)

    luminance = samples[0].mean(dim=0)  # (H, W), shallow image feature
    return torch.stack([s1, s2, s3, s4, luminance], dim=0)  # (5, H, W)


def _cache_fingerprint(dataset: CausalBenchDataset, k: int) -> dict:
    """Identifies the (dataset, k) configuration a cache was built from, so
    a stale cache at the same path (e.g. after changing the dataset's
    seed/scale/image list or k) is detected and rejected rather than
    silently reused — a config change with no visible symptom otherwise."""
    return {
        "num_hr_images": len(dataset.hr_images),
        "num_ood_patches": len(dataset.ood_patches),
        "scale": dataset.scale,
        "seed": dataset.seed,
        "dataset_len": len(dataset),
        "k": k,
    }


def precompute_signal_stacks(dataset: CausalBenchDataset, backbone, kernel_estimator, feature_bank, dino_encoder, k: int = 2, cache_path: str | None = None) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Returns a list of (signal_stack (5,H,W), label_map (H,W)) pairs,
    computed once. If cache_path is given and already exists, loads from
    it instead of recomputing (e.g. across separate training script runs)
    — but only if the cached fingerprint matches this call's dataset/k
    configuration; a mismatch raises rather than silently training on
    stale data."""
    fingerprint = _cache_fingerprint(dataset, k)
    if cache_path and os.path.exists(cache_path):
        cached = torch.load(cache_path, weights_only=True)
        if cached["fingerprint"] != fingerprint:
            raise ValueError(
                f"Cache at {cache_path} was built from a different (dataset, k) configuration "
                f"than requested: cached={cached['fingerprint']}, requested={fingerprint}. "
                "Delete the stale cache file or use a different cache_path."
            )
        return cached["items"]

    cached_items = []
    for idx in range(len(dataset)):
        lr, _hr, label_map = dataset[idx]
        signal_stack = build_signal_stack(lr, backbone, kernel_estimator, feature_bank, dino_encoder, k=k)
        cached_items.append((signal_stack, label_map))

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"fingerprint": fingerprint, "items": cached_items}, cache_path)
    return cached_items


class _CachedSignalDataset(Dataset):
    def __init__(self, cached_items: list[tuple[torch.Tensor, torch.Tensor]]):
        self.cached_items = cached_items

    def __len__(self):
        return len(self.cached_items)

    def __getitem__(self, idx):
        return self.cached_items[idx]


def train(cached_items: list[tuple[torch.Tensor, torch.Tensor]], epochs: int = 30, batch_size: int = 4, device: str = "cuda") -> FusionHead:
    loader = DataLoader(_CachedSignalDataset(cached_items), batch_size=batch_size, shuffle=True)
    model = FusionHead().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0.0
        for signal_stack, label_map in loader:
            signal_stack, label_map = signal_stack.to(device), label_map.to(device)

            optim.zero_grad()
            cause_logits, reliability = model(signal_stack)
            cause_loss = nn.functional.cross_entropy(cause_logits, label_map)
            # reliability_head has its own branch off the shared trunk (see
            # FusionHead) — without a loss term of its own, no gradient
            # ever reaches it and it stays at random init forever, despite
            # being described as a "calibrated reliability score." Target:
            # is this pixel labeled RELIABLE (per xSR-CausalBench ground
            # truth), matching the spec's "reliable/not hallucinated" framing.
            is_reliable = (label_map == CAUSE_LABELS["RELIABLE"]).float().unsqueeze(1)
            reliability_loss = nn.functional.binary_cross_entropy(reliability, is_reliable)
            loss = cause_loss + reliability_loss
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"epoch {epoch}: loss {total_loss / len(loader):.4f}")

    return model
