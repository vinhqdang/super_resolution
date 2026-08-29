import torch

from src.causalbench.build_benchmark import CAUSE_LABELS
from src.eval.ablation import (
    calibrate_equal_weight_threshold,
    calibrate_single_signal_threshold,
    predict_equal_weight,
    predict_single_signal,
)


def test_predict_single_signal_above_threshold_gives_own_cause():
    signal_map = torch.tensor([[0.9, 0.1], [0.8, 0.05]])
    pred = predict_single_signal(signal_map, threshold=0.5, signal_idx=2)
    assert pred[0, 0].item() == CAUSE_LABELS["DEGRADATION_MISMATCH"]
    assert pred[1, 0].item() == CAUSE_LABELS["DEGRADATION_MISMATCH"]
    assert pred[0, 1].item() == CAUSE_LABELS["RELIABLE"]
    assert pred[1, 1].item() == CAUSE_LABELS["RELIABLE"]


def test_predict_equal_weight_picks_dominant_signal_above_threshold():
    # 4 channels, 1x1 spatial: S3 (index 2) dominates at this pixel.
    signal_stack = torch.tensor([[[0.1]], [[0.2]], [[0.9]], [[0.05]]])
    pred = predict_equal_weight(signal_stack, threshold=0.5)
    assert pred[0, 0].item() == CAUSE_LABELS["DEGRADATION_MISMATCH"]


def test_predict_equal_weight_all_below_threshold_gives_reliable():
    signal_stack = torch.tensor([[[0.1]], [[0.2]], [[0.3]], [[0.05]]])
    pred = predict_equal_weight(signal_stack, threshold=0.5)
    assert pred[0, 0].item() == CAUSE_LABELS["RELIABLE"]


def test_predict_equal_weight_ignores_channels_beyond_the_first_four():
    # 5th channel (e.g. luminance) has the largest raw value but must be ignored.
    signal_stack = torch.tensor([[[0.1]], [[0.2]], [[0.9]], [[0.05]], [[0.99]]])
    pred = predict_equal_weight(signal_stack, threshold=0.5)
    assert pred[0, 0].item() == CAUSE_LABELS["DEGRADATION_MISMATCH"]


def _synthetic_cached_items(signal_idx: int, cause: int, n: int = 20):
    """Builds a calibration set where signal `signal_idx` cleanly separates
    `cause` pixels (signal ~0.9) from RELIABLE pixels (signal ~0.1), so a
    correct calibration should find a threshold that gets ~100% accuracy."""
    items = []
    for i in range(n):
        signal_stack = torch.full((4, 4, 4), 0.1)
        label_map = torch.full((4, 4), CAUSE_LABELS["RELIABLE"], dtype=torch.long)
        signal_stack[signal_idx, :2, :] = 0.9
        label_map[:2, :] = cause
        items.append((signal_stack, label_map))
    return items


def test_calibrate_single_signal_threshold_separates_cleanly_when_possible():
    cached_items = _synthetic_cached_items(signal_idx=1, cause=CAUSE_LABELS["PRIOR_RELIANCE"])
    threshold = calibrate_single_signal_threshold(cached_items, signal_idx=1)
    acc = 0
    total = 0
    for signal_stack, label_map in cached_items:
        pred = predict_single_signal(signal_stack[1], threshold, signal_idx=1)
        acc += int((pred == label_map).sum().item())
        total += label_map.numel()
    assert acc / total > 0.99


def test_calibrate_equal_weight_threshold_separates_cleanly_when_possible():
    cached_items = _synthetic_cached_items(signal_idx=3, cause=CAUSE_LABELS["DISTRIBUTION_SHIFT"])
    threshold = calibrate_equal_weight_threshold(cached_items)
    acc = 0
    total = 0
    for signal_stack, label_map in cached_items:
        pred = predict_equal_weight(signal_stack, threshold)
        acc += int((pred == label_map).sum().item())
        total += label_map.numel()
    assert acc / total > 0.99
