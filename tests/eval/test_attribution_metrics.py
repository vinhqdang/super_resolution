import pytest
import torch

from src.eval.attribution_metrics import mean_iou, pixel_accuracy


def test_pixel_accuracy_perfect_match():
    pred = torch.tensor([[0, 1], [2, 3]])
    gt = torch.tensor([[0, 1], [2, 3]])
    assert pixel_accuracy(pred, gt) == 1.0


def test_pixel_accuracy_no_match():
    pred = torch.tensor([[0, 0], [0, 0]])
    gt = torch.tensor([[1, 1], [1, 1]])
    assert pixel_accuracy(pred, gt) == 0.0


def test_mean_iou_perfect_match_is_one():
    pred = torch.randint(0, 5, (16, 16))
    gt = pred.clone()
    assert mean_iou(pred, gt, num_classes=5) == 1.0


def test_mean_iou_no_overlap_is_zero():
    pred = torch.zeros(4, 4, dtype=torch.long)
    gt = torch.ones(4, 4, dtype=torch.long)
    assert mean_iou(pred, gt, num_classes=5) == 0.0


def test_mean_iou_realistic_multiclass_partial_overlap():
    # gt: class0 x4, class1 x2, class2 x2. pred: partially overlapping,
    # plus a spurious class-3 prediction absent from gt entirely.
    gt = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
    pred = torch.tensor([0, 0, 1, 1, 1, 1, 2, 3])
    # class0: gt={0,1,2,3}, pred={0,1} -> inter=2, union=4 -> IoU=0.5
    # class1: gt={4,5}, pred={2,3,4,5} -> inter=2, union=4 -> IoU=0.5
    # class2: gt={6,7}, pred={6} -> inter=1, union=2 -> IoU=0.5
    # class3: gt={}, pred={7} -> inter=0, union=1 -> IoU=0.0 (not skipped: union != 0)
    # class4: absent from both -> skipped
    expected = (0.5 + 0.5 + 0.5 + 0.0) / 4
    assert mean_iou(pred, gt, num_classes=5) == pytest.approx(expected)
