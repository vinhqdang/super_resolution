"""Trains the fusion head on xSR-CausalBench. Each dataset item's signal
stack requires ~5 diffusion sampling calls per k (1 hop-1 call + 4 tiled
hop-2 calls) through the backbone (Task 5) — expensive enough that
recomputing it on every training epoch would make Phase 1 training take
days on an 8GB laptop GPU. `precompute_signal_stacks` therefore builds
every (signal_stack, label_map) pair ONCE, optionally caching to disk, and
`train` iterates epochs over that cheap, cached tensor set — only the
small fusion head does repeated forward/backward passes.
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.backbone.diffusion_backbone import estimate_prior_reliance_map
from src.causalbench.dataset import CausalBenchDataset
from src.fusion.model import FusionHead
from src.signals.degradation_mismatch import compute_degradation_mismatch
from src.signals.distribution_shift import FeatureBank, extract_dinov2_features
from src.signals.null_space import compute_null_space_variance


def build_signal_stack(lr, backbone, kernel_estimator, feature_bank, dino_encoder, k=2):
    samples = backbone.sample_k_16x(lr, k=k, base_seed=0)
    s1 = compute_null_space_variance(samples)  # (H, W)

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


def precompute_signal_stacks(dataset: CausalBenchDataset, backbone, kernel_estimator, feature_bank, dino_encoder, k: int = 2, cache_path: str | None = None) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Returns a list of (signal_stack (5,H,W), label_map (H,W)) pairs,
    computed once. If cache_path is given and already exists, loads from
    it instead of recomputing (e.g. across separate training script runs)."""
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)

    cached_items = []
    for idx in range(len(dataset)):
        lr, _hr, label_map = dataset[idx]
        signal_stack = build_signal_stack(lr, backbone, kernel_estimator, feature_bank, dino_encoder, k=k)
        cached_items.append((signal_stack, label_map))

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cached_items, cache_path)
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
            cause_logits, _ = model(signal_stack)
            loss = nn.functional.cross_entropy(cause_logits, label_map)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"epoch {epoch}: loss {total_loss / len(loader):.4f}")

    return model
