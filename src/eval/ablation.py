"""Ablation baselines (spec Section 5, item 3): does the learned fusion
head add value over (a) each raw causal signal used alone and (b) an
untrained equal-weight combination of all four signals? Both baseline
families are threshold rules with no learned weights; their thresholds
are calibrated by grid search on a labeled calibration set (the fusion
head's own training set, via src.fusion.train.precompute_signal_stacks's
cache) and then applied, frozen, to a separate held-out set — never fit
and evaluated on the same data.
"""
import torch

from src.causalbench.build_benchmark import CAUSE_LABELS

_RELIABLE = CAUSE_LABELS["RELIABLE"]
_SIGNAL_TO_CAUSE = [
    CAUSE_LABELS["ILL_POSED"],
    CAUSE_LABELS["PRIOR_RELIANCE"],
    CAUSE_LABELS["DEGRADATION_MISMATCH"],
    CAUSE_LABELS["DISTRIBUTION_SHIFT"],
]  # signal_stack channel i's own cause, matching xSR-CausalBench's construction order


def predict_single_signal(signal_map: torch.Tensor, threshold: float, signal_idx: int) -> torch.Tensor:
    """signal_map: (H, W), one raw signal channel. Predicts this signal's
    own cause where it exceeds threshold, RELIABLE elsewhere — a single
    raw signal alone cannot distinguish among the other three causes."""
    cause = _SIGNAL_TO_CAUSE[signal_idx]
    cause_map = torch.full_like(signal_map, cause, dtype=torch.long)
    reliable_map = torch.full_like(signal_map, _RELIABLE, dtype=torch.long)
    return torch.where(signal_map > threshold, cause_map, reliable_map)


def predict_equal_weight(signal_stack: torch.Tensor, threshold: float) -> torch.Tensor:
    """signal_stack: (>=4, H, W); only the first 4 channels (S1-S4) are
    used. Predicts whichever raw signal is largest at that pixel, mapped
    to its own cause, where that max exceeds threshold; RELIABLE
    elsewhere. No learned weighting — "whichever signal is loudest"."""
    max_vals, max_idx = signal_stack[:4].max(dim=0)  # (H, W), (H, W)
    cause_lookup = torch.tensor(_SIGNAL_TO_CAUSE, dtype=torch.long, device=signal_stack.device)
    cause_map = cause_lookup[max_idx]
    reliable_map = torch.full_like(cause_map, _RELIABLE)
    return torch.where(max_vals > threshold, cause_map, reliable_map)


def _pixel_accuracy_at_threshold(cached_items, threshold: float, predict_fn) -> float:
    correct, total = 0, 0
    for signal_stack, label_map in cached_items:
        pred = predict_fn(signal_stack, threshold)
        # label_map comes straight from CausalBenchDataset (CPU), while
        # signal_stack (and therefore pred) is whatever device the cache
        # was built on — CUDA when precompute_signal_stacks ran against the
        # real backbone. Synthetic same-device tensors in this file's own
        # unit tests never exercised this mismatch.
        correct += int((pred == label_map.to(pred.device)).sum().item())
        total += label_map.numel()
    return correct / total if total > 0 else 0.0


def calibrate_single_signal_threshold(cached_items, signal_idx: int, grid: torch.Tensor | None = None) -> float:
    """cached_items: list of (signal_stack (>=4,H,W), label_map (H,W))
    pairs, e.g. from src.fusion.train.precompute_signal_stacks's cache.
    Grid-searches the threshold that maximizes pixel accuracy of
    predict_single_signal on this set (intended to be a calibration/
    training set, evaluated separately on held-out data)."""
    if grid is None:
        grid = torch.linspace(0.0, 1.0, steps=51)
    best_threshold, best_acc = 0.5, -1.0
    for t in grid.tolist():
        acc = _pixel_accuracy_at_threshold(cached_items, t, lambda s, th: predict_single_signal(s[signal_idx], th, signal_idx))
        if acc > best_acc:
            best_acc, best_threshold = acc, t
    return best_threshold


def calibrate_equal_weight_threshold(cached_items, grid: torch.Tensor | None = None) -> float:
    """Same grid-search calibration as calibrate_single_signal_threshold,
    for the equal-weight combination rule."""
    if grid is None:
        grid = torch.linspace(0.0, 1.0, steps=51)
    best_threshold, best_acc = 0.5, -1.0
    for t in grid.tolist():
        acc = _pixel_accuracy_at_threshold(cached_items, t, lambda s, th: predict_equal_weight(s, th))
        if acc > best_acc:
            best_acc, best_threshold = acc, t
    return best_threshold
