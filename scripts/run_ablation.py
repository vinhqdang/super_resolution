"""Ablation study (spec Section 5, item 3; named in manuscript/main.tex
Section 5.4 as the single most important missing evaluation piece): does
the trained fusion head add value over (a) each raw causal signal used
alone and (b) an untrained equal-weight combination of all four signals?

Requires the exact same artifacts as scripts/run_evaluation.py, PLUS the
signal-stack cache scripts/train_fusion_head.py wrote at
checkpoints/causalbench_signal_cache.pt (used here as the calibration set
for the two baseline families' thresholds — reusing it costs zero extra
diffusion sampling, since it was already computed once during fusion-head
training). If that cache is missing, this script raises with a clear
message rather than silently recomputing it (which would cost the same
~2-3 hours logged for the original precompute pass).

Threshold calibration and held-out evaluation are on disjoint data: the
cache is the fusion head's own training set (DIV2K valid [:40] + ood
[40:50]); the held-out set here is the same range scripts/run_evaluation.py
uses (DIV2K valid [50:70] + ood [70:75]), matching that script exactly so
the "ours" row in this comparison table reproduces run_evaluation.py's own
attribution-accuracy number.

Run: conda run -n py313 python scripts/run_ablation.py
"""
import glob
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

from src.backbone.diffusion_backbone import DEFAULT_MODEL_ID, DiffusionSRBackbone
from src.causalbench.dataset import CausalBenchDataset
from src.eval.ablation import (
    calibrate_equal_weight_threshold,
    calibrate_single_signal_threshold,
    predict_equal_weight,
    predict_single_signal,
)
from src.eval.attribution_metrics import mean_iou, pixel_accuracy
from src.fusion.infer import run_chasr
from src.fusion.model import FusionHead
from src.fusion.train import SIGNAL_STACK_K
from src.signals.distribution_shift import FeatureBank
from src.signals.kernel_estimator import KernelEstimator

HR_PATCH_SIZE = 1024  # must match DiffusionSRBackbone's fixed 16x output size
EVAL_MAX_IMAGES = 20
MODEL_ID = os.environ.get("CHASR_MODEL_ID", DEFAULT_MODEL_ID)
CACHE_PATH = "checkpoints/causalbench_signal_cache.pt"
SIGNAL_NAMES = ["S1 (ill-posedness)", "S2 (prior-reliance)", "S3 (mismatch)", "S4 (distribution-shift)"]


def main():
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(
            f"{CACHE_PATH} not found. Run scripts/train_fusion_head.py first (it writes this cache "
            "as a side effect of the one-time signal-stack precompute pass) before running the ablation."
        )
    cached = torch.load(CACHE_PATH, weights_only=True)
    calibration_items = cached["items"]

    print("Calibrating baseline thresholds on the fusion head's training set...")
    equal_weight_threshold = calibrate_equal_weight_threshold(calibration_items)
    single_signal_thresholds = [calibrate_single_signal_threshold(calibration_items, signal_idx=i) for i in range(4)]
    print(f"  Equal-weight threshold: {equal_weight_threshold:.3f}")
    for name, t in zip(SIGNAL_NAMES, single_signal_thresholds):
        print(f"  {name} threshold: {t:.3f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    results = {
        "Fusion head (ours)": {"accuracy": [], "iou": []},
        "Equal-weight combination": {"accuracy": [], "iou": []},
    }
    for name in SIGNAL_NAMES:
        results[f"{name} alone"] = {"accuracy": [], "iou": []}

    print(f"\nEvaluating on {len(held_out)} held-out xSR-CausalBench items...")
    for i in range(len(held_out)):
        lr, _hr, gt_label_map = held_out[i]
        result = run_chasr(lr, backbone, kernel_estimator, feature_bank, dino_encoder, fusion_head, k=SIGNAL_STACK_K)

        cause_map_cpu = result["cause_map"].cpu()
        signal_stack_cpu = result["signal_stack"].cpu().float()

        results["Fusion head (ours)"]["accuracy"].append(pixel_accuracy(cause_map_cpu, gt_label_map))
        results["Fusion head (ours)"]["iou"].append(mean_iou(cause_map_cpu, gt_label_map))

        equal_weight_pred = predict_equal_weight(signal_stack_cpu, equal_weight_threshold)
        results["Equal-weight combination"]["accuracy"].append(pixel_accuracy(equal_weight_pred, gt_label_map))
        results["Equal-weight combination"]["iou"].append(mean_iou(equal_weight_pred, gt_label_map))

        for idx, name in enumerate(SIGNAL_NAMES):
            single_pred = predict_single_signal(signal_stack_cpu[idx], single_signal_thresholds[idx], signal_idx=idx)
            results[f"{name} alone"]["accuracy"].append(pixel_accuracy(single_pred, gt_label_map))
            results[f"{name} alone"]["iou"].append(mean_iou(single_pred, gt_label_map))

    print("\n=== Ablation results (per-image macro-average, held-out set) ===")
    print(f"{'Method':<28} {'Pixel accuracy':>15} {'Mean IoU':>10}")
    for name, metrics in results.items():
        acc = np.mean(metrics["accuracy"])
        iou = np.mean(metrics["iou"])
        print(f"{name:<28} {acc:>14.4f} {iou:>10.4f}")


if __name__ == "__main__":
    main()
