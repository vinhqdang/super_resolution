"""Runs the full Phase 1 evaluation protocol (spec Section 5) and prints a
results table: fidelity metrics (PSNR/SSIM/LPIPS), attribution accuracy/mIoU
(both per-image macro-average and dataset-level accumulated-confusion) with
a majority-class baseline, per-metric spread across the held-out set, and
the disentanglement correlation matrix. Also saves two qualitative
LR/SR/ground-truth/predicted-attribution examples per xSR-CausalBench
procedure to manuscript/figures/ (see below), for the manuscript's
qualitative figures.

HR images are resized to 1024x1024, matching scripts/train_fusion_head.py
(see that file's docstring for why — must match the backbone's fixed 16x
factor). The held-out image range [50:70] is deliberately disjoint from
train_fusion_head.py's [0:40] + [40:50] ood range so evaluation never
sees a training image.

For each procedure, two qualitative examples are saved rather than one:
the held-out item with the highest per-image pixel accuracy for that
procedure ("ex1", a representative success case) and the item with the
lowest ("ex2", a representative hard case). This selection is by a fixed,
pre-registered criterion (per-image accuracy, computed identically to the
paper's own headline metric) rather than hand-picked after the fact for
visual appeal, so the two examples per procedure are free to look
different — including both "recovers the region cleanly" and "misses it
almost entirely" — without that being evidence of cherry-picking.

mIoU aggregation: this reports both the per-image macro-average (mean of
mean_iou() calls, one per image — matches the Task 13 plan's mean_iou()
signature, which operates on one image at a time) and the dataset-level
mIoU (accumulate one confusion matrix over the whole held-out set, then
divide per class). These are not numerically equivalent, and the
literature default for reporting mIoU is dataset-level; both are reported
so neither number is presented without its aggregation convention named.

Majority-class baseline: xSR-CausalBench's four procedures do not produce
a uniform class prior (two procedures label the whole image with one
class; two label only a small sub-region and leave the rest RELIABLE), so
a trivial always-predict-the-majority-class rule is a stronger reference
point than 1/num_classes for judging whether the fusion head has learned
anything beyond the class prior.

Run: conda run -n py313 python scripts/run_evaluation.py
"""
import glob
import os

import lpips
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

from src.backbone.diffusion_backbone import DEFAULT_MODEL_ID, DiffusionSRBackbone
from src.causalbench.build_benchmark import CAUSE_LABELS
from src.causalbench.dataset import CausalBenchDataset
from src.eval.attribution_metrics import mean_iou, pixel_accuracy
from src.eval.disentanglement import signal_correlation_matrix
from src.eval.fidelity_metrics import compute_lpips, compute_psnr, compute_ssim
from src.fusion.infer import run_chasr
from src.fusion.model import FusionHead
from src.fusion.train import SIGNAL_STACK_K
from src.signals.distribution_shift import FeatureBank
from src.signals.kernel_estimator import KernelEstimator

HR_PATCH_SIZE = 1024  # must match DiffusionSRBackbone's fixed 16x output size
EVAL_MAX_IMAGES = 20
MODEL_ID = os.environ.get("CHASR_MODEL_ID", DEFAULT_MODEL_ID)
NUM_CLASSES = 5
FIGURE_DIR = "manuscript/figures"
_PROCEDURES = ["ill_posed", "prior_reliance", "mismatch", "distribution_shift"]  # matches src/causalbench/dataset.py's ordering

_CAUSE_COLORS = {  # RGB; used only for the qualitative attribution-map figure
    0: (230, 25, 75),    # ILL_POSED
    1: (60, 180, 75),    # PRIOR_RELIANCE
    2: (255, 225, 25),   # DEGRADATION_MISMATCH
    3: (0, 130, 200),    # DISTRIBUTION_SHIFT
    4: (128, 128, 128),  # RELIABLE
}


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    out = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for cls, color in _CAUSE_COLORS.items():
        out[label_map == cls] = color
    return out


def accumulate_confusion(pred: torch.Tensor, gt: torch.Tensor, confusion: np.ndarray) -> None:
    """confusion[g, p] += pixel count with ground truth g predicted as p."""
    for p in range(NUM_CLASSES):
        pred_mask = pred == p
        for g in range(NUM_CLASSES):
            confusion[g, p] += int((pred_mask & (gt == g)).sum().item())


def dataset_level_miou(confusion: np.ndarray) -> float:
    ious = []
    for c in range(NUM_CLASSES):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        union = tp + fp + fn
        if union == 0:
            continue  # class absent from both prediction and ground truth across the whole held-out set
        ious.append(tp / union)
    return float(sum(ious) / len(ious)) if ious else 0.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(FIGURE_DIR, exist_ok=True)

    hr_paths = sorted(glob.glob("data/DIV2K_valid_HR/*.png"))[50 : 50 + EVAL_MAX_IMAGES]
    hr_images = [np.asarray(Image.open(p).convert("RGB").resize((HR_PATCH_SIZE, HR_PATCH_SIZE))).astype(np.float32) / 255.0 for p in hr_paths]
    ood_paths = sorted(glob.glob("data/DIV2K_valid_HR/*.png"))[50 + EVAL_MAX_IMAGES : 50 + EVAL_MAX_IMAGES + 5]
    ood_patches = [np.asarray(Image.open(p).convert("RGB").resize((64, 64))).astype(np.float32) / 255.0 for p in ood_paths]

    held_out = CausalBenchDataset(hr_images, ood_patches, scale=16, seed=100)

    kernel_estimator = KernelEstimator().to(device)
    kernel_estimator.load_state_dict(torch.load("checkpoints/kernel_estimator.pt", weights_only=True))
    kernel_estimator.eval()
    feature_bank = FeatureBank(torch.load("checkpoints/feature_bank.pt", weights_only=True))
    dino_encoder = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    backbone = DiffusionSRBackbone(device=device, model_id=MODEL_ID)
    fusion_head = FusionHead().to(device)
    fusion_head.load_state_dict(torch.load("checkpoints/fusion_head.pt", weights_only=True))
    fusion_head.eval()
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    psnrs, ssims, lpipses, accuracies, ious, all_signals = [], [], [], [], [], []
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    gt_class_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    procedure_candidates = {p: [] for p in _PROCEDURES}  # procedure -> list of (per_item_accuracy, lr_np, sr_np, gt_np, pred_np)

    for i in range(len(held_out)):
        lr, hr, gt_label_map = held_out[i]
        procedure = _PROCEDURES[i % len(_PROCEDURES)]
        result = run_chasr(lr, backbone, kernel_estimator, feature_bank, dino_encoder, fusion_head, k=SIGNAL_STACK_K)

        # .cpu().float(): sr_image comes straight from the real backbone
        # (CUDA fp16), which .numpy() cannot convert directly — the stub
        # backbone used in tests/fusion/test_infer.py returns CPU float32,
        # so this path was never exercised until running against real data.
        sr_np = result["sr_image"].cpu().float().permute(1, 2, 0).clamp(0, 1).numpy()
        hr_np = hr.permute(1, 2, 0).clamp(0, 1).numpy()
        psnrs.append(compute_psnr(sr_np, hr_np))
        ssims.append(compute_ssim(sr_np, hr_np))

        sr_for_lpips = result["sr_image"].float().clamp(0, 1)
        hr_for_lpips = hr.to(sr_for_lpips.device)
        lpipses.append(compute_lpips(sr_for_lpips, hr_for_lpips, lpips_model))

        # .cpu(): cause_map comes from the real (CUDA) fusion_head forward
        # pass while gt_label_map is CPU (straight from the Dataset) —
        # comparing tensors on different devices raises, and this path
        # (like sr_image above) was only ever exercised via CPU stubs before.
        cause_map_cpu = result["cause_map"].cpu()
        item_accuracy = pixel_accuracy(cause_map_cpu, gt_label_map)
        accuracies.append(item_accuracy)
        ious.append(mean_iou(cause_map_cpu, gt_label_map))
        accumulate_confusion(cause_map_cpu, gt_label_map, confusion)
        for cls in range(NUM_CLASSES):
            gt_class_pixel_counts[cls] += int((gt_label_map == cls).sum().item())

        # .float(): guards the disentanglement correlation matmul below
        # against fp16 precision loss/overflow when accumulating across
        # ~20M elements (EVAL_MAX_IMAGES x 1024x1024) on hardware without
        # fp32 tensor-core accumulation.
        all_signals.append(result["signal_stack"].reshape(4, -1).cpu().float())  # (4, H*W) per image

        lr_np = lr.permute(1, 2, 0).clamp(0, 1).numpy()
        lr_thumb = np.asarray(Image.fromarray((lr_np * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST))
        procedure_candidates[procedure].append((item_accuracy, lr_thumb, (sr_np * 255).astype(np.uint8), colorize_label_map(gt_label_map.numpy()), colorize_label_map(cause_map_cpu.numpy())))

    for procedure, candidates in procedure_candidates.items():
        candidates.sort(key=lambda c: c[0])  # ascending per-item accuracy: worst first, best last
        worst, best = candidates[0], candidates[-1]
        for suffix, (_, lr_arr, sr_arr, gt_arr, pred_arr) in [("ex1", best), ("ex2", worst)]:
            Image.fromarray(lr_arr).save(f"{FIGURE_DIR}/{procedure}_{suffix}_lr.png")
            Image.fromarray(sr_arr).save(f"{FIGURE_DIR}/{procedure}_{suffix}_sr.png")
            Image.fromarray(gt_arr).save(f"{FIGURE_DIR}/{procedure}_{suffix}_gt.png")
            Image.fromarray(pred_arr).save(f"{FIGURE_DIR}/{procedure}_{suffix}_pred.png")
        print(f"{procedure}: ex1 (best) per-image accuracy = {best[0]:.4f}, ex2 (worst) per-image accuracy = {worst[0]:.4f}")

    print(f"PSNR: {np.mean(psnrs):.2f} (std {np.std(psnrs):.2f})")
    print(f"SSIM: {np.mean(ssims):.4f} (std {np.std(ssims):.4f})")
    print(f"LPIPS: {np.mean(lpipses):.4f} (std {np.std(lpipses):.4f})")
    print(f"Attribution pixel accuracy: {np.mean(accuracies):.4f} (std {np.std(accuracies):.4f})")
    print(f"Attribution mIoU, per-image macro-average: {np.mean(ious):.4f} (std {np.std(ious):.4f})")
    print(f"Attribution mIoU, dataset-level accumulated confusion: {dataset_level_miou(confusion):.4f}")

    majority_class = int(np.argmax(gt_class_pixel_counts))
    majority_frac = float(gt_class_pixel_counts[majority_class] / gt_class_pixel_counts.sum())
    class_names = list(CAUSE_LABELS.keys())
    print(f"Ground-truth pixel distribution: {dict(zip(class_names, gt_class_pixel_counts.tolist()))}")
    print(f"Majority-class baseline ('{class_names[majority_class]}' everywhere): accuracy = {majority_frac:.4f}")

    flattened_signals = torch.cat(all_signals, dim=1)  # (4, N * H * W) across all held-out images
    corr = signal_correlation_matrix(flattened_signals)
    print("S1-S4 disentanglement correlation matrix (spec Section 5, item 5):")
    print(corr)


if __name__ == "__main__":
    main()
