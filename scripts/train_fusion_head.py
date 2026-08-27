"""Full Phase 1 fusion-head training entry point. Requires:
- checkpoints/kernel_estimator.pt (Task 8's scripts/train_kernel_estimator.py)
- checkpoints/feature_bank.pt (Task 10's scripts/build_feature_bank.py)
- data/DIV2K_train_HR/, data/DIV2K_valid_HR/ (scripts/download_datasets.py)

HR images are resized to exactly 1024x1024 — this must match
DiffusionSRBackbone.sample_k_16x's fixed 16x factor (Task 5): at
scale=16, CausalBenchDataset then produces exactly 64x64 LR patches,
which is what the backbone expects as input. Do not change one without
the other. TRAIN_MAX_IMAGES/TRAIN_K are kept small deliberately — each
dataset item costs ~5 diffusion calls per k during the one-time
precompute pass below; 40 images x 4 procedures x k=2 x 5 calls = 1600
diffusion calls, roughly 2-3 hours one-time on an 8GB laptop GPU. Raise
these only once Phase 1 numbers exist and you're deliberately scaling up.

Run: conda run -n py313 python scripts/train_fusion_head.py
"""
import glob
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

from src.backbone.diffusion_backbone import DiffusionSRBackbone
from src.causalbench.dataset import CausalBenchDataset
from src.fusion.train import SIGNAL_STACK_K, precompute_signal_stacks, train
from src.signals.distribution_shift import FeatureBank
from src.signals.kernel_estimator import KernelEstimator

HR_PATCH_SIZE = 1024  # must match DiffusionSRBackbone's fixed 16x output size
TRAIN_MAX_IMAGES = 40
TRAIN_K = SIGNAL_STACK_K  # must match run_chasr's default k (src/fusion/infer.py) — see that constant's docstring


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hr_paths = sorted(glob.glob("data/DIV2K_valid_HR/*.png"))[:TRAIN_MAX_IMAGES]
    hr_images = [np.asarray(Image.open(p).convert("RGB").resize((HR_PATCH_SIZE, HR_PATCH_SIZE))).astype(np.float32) / 255.0 for p in hr_paths]
    ood_paths = sorted(glob.glob("data/DIV2K_valid_HR/*.png"))[TRAIN_MAX_IMAGES : TRAIN_MAX_IMAGES + 10]
    ood_patches = [np.asarray(Image.open(p).convert("RGB").resize((64, 64))).astype(np.float32) / 255.0 for p in ood_paths]

    dataset = CausalBenchDataset(hr_images, ood_patches, scale=16, seed=0)

    kernel_estimator = KernelEstimator().to(device)
    kernel_estimator.load_state_dict(torch.load("checkpoints/kernel_estimator.pt", weights_only=True))
    kernel_estimator.eval()

    feature_bank = FeatureBank(torch.load("checkpoints/feature_bank.pt", weights_only=True))
    dino_encoder = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    backbone = DiffusionSRBackbone(device=device)

    cached_items = precompute_signal_stacks(
        dataset, backbone, kernel_estimator, feature_bank, dino_encoder, k=TRAIN_K, cache_path="checkpoints/causalbench_signal_cache.pt"
    )
    model = train(cached_items, epochs=30, device=device)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/fusion_head.pt")
    print("Saved checkpoints/fusion_head.pt")


if __name__ == "__main__":
    main()
