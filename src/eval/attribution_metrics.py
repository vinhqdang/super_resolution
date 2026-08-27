"""Attribution accuracy metrics (spec Section 5, item 3 — the headline
result): does the fusion head's predicted dominant cause match the
synthetic ground truth on xSR-CausalBench?
"""
import torch


def pixel_accuracy(pred_labels: torch.Tensor, gt_labels: torch.Tensor) -> float:
    return float((pred_labels == gt_labels).float().mean().item())


def mean_iou(pred_labels: torch.Tensor, gt_labels: torch.Tensor, num_classes: int = 5) -> float:
    ious = []
    for cls in range(num_classes):
        pred_mask = pred_labels == cls
        gt_mask = gt_labels == cls
        union = (pred_mask | gt_mask).sum().item()
        if union == 0:
            continue  # class absent from both — skip rather than penalize
        intersection = (pred_mask & gt_mask).sum().item()
        ious.append(intersection / union)
    return float(sum(ious) / len(ious)) if ious else 0.0
