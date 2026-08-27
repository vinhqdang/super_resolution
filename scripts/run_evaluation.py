"""Runs the full Phase 1 evaluation protocol (spec Section 5) and prints a
results table: fidelity metrics, attribution accuracy/mIoU on a held-out
xSR-CausalBench split, and the disentanglement correlation matrix.

HR images are resized to 1024x1024, matching scripts/train_fusion_head.py
(see that file's docstring for why — must match the backbone's fixed 16x
factor). The held-out image range [50:70] is deliberately disjoint from
train_fusion_head.py's [0:40] + [40:50] ood range so evaluation never
sees a training image.

mIoU aggregation: this reports the mean of per-image mean_iou() calls
(macro-average over images), not the standard dataset-level mIoU
(accumulate one confusion matrix over the whole held-out set, then divide
per class). These are not numerically equivalent, and the literature
default for reporting mIoU is dataset-level. Per-image macro-averaging
was kept here to match the Task 13 plan's mean_iou() signature (which
operates on one image at a time) without introducing a second aggregation
path; if this becomes the reported headline number for the manuscript,
switch to dataset-level accumulation (or report both) rather than relying
on this having been a considered choice already.

Run: conda run -n py313 python scripts/run_evaluation.py
"""
import glob
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

from src.backbone.diffusion_backbone import DEFAULT_MODEL_ID, DiffusionSRBackbone
from src.causalbench.dataset import CausalBenchDataset
from src.eval.attribution_metrics import mean_iou, pixel_accuracy
from src.eval.disentanglement import signal_correlation_matrix
from src.eval.fidelity_metrics import compute_psnr, compute_ssim
from src.fusion.infer import run_chasr
from src.fusion.model import FusionHead
from src.fusion.train import SIGNAL_STACK_K
from src.signals.distribution_shift import FeatureBank
from src.signals.kernel_estimator import KernelEstimator

HR_PATCH_SIZE = 1024  # must match DiffusionSRBackbone's fixed 16x output size
EVAL_MAX_IMAGES = 20
MODEL_ID = os.environ.get("CHASR_MODEL_ID", DEFAULT_MODEL_ID)


def main():
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

    psnrs, ssims, accuracies, ious, all_signals = [], [], [], [], []
    for i in range(len(held_out)):
        lr, hr, gt_label_map = held_out[i]
        # k explicit (not relying on run_chasr's default): must match the K
        # the fusion head was actually trained on (scripts/train_fusion_head.py's
        # TRAIN_K) — see SIGNAL_STACK_K's docstring in src/fusion/train.py.
        result = run_chasr(lr, backbone, kernel_estimator, feature_bank, dino_encoder, fusion_head, k=SIGNAL_STACK_K)

        # .cpu().float(): sr_image comes straight from the real backbone
        # (CUDA fp16), which .numpy() cannot convert directly — the stub
        # backbone used in tests/fusion/test_infer.py returns CPU float32,
        # so this path was never exercised until running against real data.
        sr_np = result["sr_image"].cpu().float().permute(1, 2, 0).clamp(0, 1).numpy()
        hr_np = hr.permute(1, 2, 0).clamp(0, 1).numpy()
        psnrs.append(compute_psnr(sr_np, hr_np))
        ssims.append(compute_ssim(sr_np, hr_np))

        # .cpu(): cause_map comes from the real (CUDA) fusion_head forward
        # pass while gt_label_map is CPU (straight from the Dataset) —
        # comparing tensors on different devices raises, and this path
        # (like sr_image above) was only ever exercised via CPU stubs before.
        cause_map_cpu = result["cause_map"].cpu()
        accuracies.append(pixel_accuracy(cause_map_cpu, gt_label_map))
        ious.append(mean_iou(cause_map_cpu, gt_label_map))

        # .float(): guards the disentanglement correlation matmul below
        # against fp16 precision loss/overflow when accumulating across
        # ~20M elements (EVAL_MAX_IMAGES x 1024x1024) on hardware without
        # fp32 tensor-core accumulation.
        all_signals.append(result["signal_stack"].reshape(4, -1).cpu().float())  # (4, H*W) per image

    print(f"PSNR: {np.mean(psnrs):.2f}")
    print(f"SSIM: {np.mean(ssims):.4f}")
    print(f"Attribution pixel accuracy: {np.mean(accuracies):.4f}")
    print(f"Attribution mIoU: {np.mean(ious):.4f}")

    flattened_signals = torch.cat(all_signals, dim=1)  # (4, N * H * W) across all held-out images
    corr = signal_correlation_matrix(flattened_signals)
    print("S1-S4 disentanglement correlation matrix (spec Section 5, item 5):")
    print(corr)


if __name__ == "__main__":
    main()
