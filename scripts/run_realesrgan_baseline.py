"""Real-ESRGAN fidelity baseline (RRDBNet x4plus, chained twice for 16x),
run on the exact same held-out xSR-CausalBench items `run_evaluation.py`
uses for its own PSNR/SSIM/LPIPS numbers (same `CausalBenchDataset`
construction, same seed, same 20 DIV2K images x 4 procedures = 80 items),
so Table 1's fidelity numbers have a real, same-data comparison point
rather than only the frozen diffusion backbone's own numbers. This is a
fidelity-only comparison: Real-ESRGAN has no causal signals and is not
part of CHASR's attribution pipeline, so no attribution/mIoU numbers apply.

Run: conda run -n py313 python scripts/run_realesrgan_baseline.py
"""
import glob

import lpips
import numpy as np
import torch
from PIL import Image

from src.baselines.real_esrgan import load_real_esrgan_x4plus, upscale_16x
from src.causalbench.dataset import CausalBenchDataset
from src.eval.fidelity_metrics import compute_lpips, compute_psnr, compute_ssim

HR_PATCH_SIZE = 1024  # must match run_evaluation.py
EVAL_MAX_IMAGES = 20


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_real_esrgan_x4plus(device=device)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    hr_paths = sorted(glob.glob("data/DIV2K_valid_HR/*.png"))[50 : 50 + EVAL_MAX_IMAGES]
    hr_images = [np.asarray(Image.open(p).convert("RGB").resize((HR_PATCH_SIZE, HR_PATCH_SIZE))).astype(np.float32) / 255.0 for p in hr_paths]
    ood_paths = sorted(glob.glob("data/DIV2K_valid_HR/*.png"))[50 + EVAL_MAX_IMAGES : 50 + EVAL_MAX_IMAGES + 5]
    ood_patches = [np.asarray(Image.open(p).convert("RGB").resize((64, 64))).astype(np.float32) / 255.0 for p in ood_paths]

    held_out = CausalBenchDataset(hr_images, ood_patches, scale=16, seed=100)

    psnrs, ssims, lpipses = [], [], []
    for i in range(len(held_out)):
        lr, hr, _gt_label_map = held_out[i]
        sr = upscale_16x(model, lr)

        sr_np = sr.permute(1, 2, 0).clamp(0, 1).numpy()
        hr_np = hr.permute(1, 2, 0).clamp(0, 1).numpy()
        psnrs.append(compute_psnr(sr_np, hr_np))
        ssims.append(compute_ssim(sr_np, hr_np))
        lpipses.append(compute_lpips(sr.to(device), hr.to(device), lpips_model))

    print(f"Real-ESRGAN (chained x4 twice, 16x), {len(held_out)} held-out items: PSNR {np.mean(psnrs):.2f} (std {np.std(psnrs):.2f})")
    print(f"Real-ESRGAN (chained x4 twice, 16x), {len(held_out)} held-out items: SSIM {np.mean(ssims):.4f} (std {np.std(ssims):.4f})")
    print(f"Real-ESRGAN (chained x4 twice, 16x), {len(held_out)} held-out items: LPIPS {np.mean(lpipses):.4f} (std {np.std(lpipses):.4f})")


if __name__ == "__main__":
    main()
