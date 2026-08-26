"""Manual integration check: downloads the fp16 variant of
stabilityai/stable-diffusion-x4-upscaler (~1.7GB) and runs one real 16x
chained upscale. Not run in CI/pytest — run by hand with:
conda run -n py313 python scripts/smoke_test_backbone.py

Set CHASR_MODEL_ID to a local directory (same layout as the hub repo) to
load a pre-downloaded snapshot instead — useful if huggingface_hub's own
downloader stalls on a given network even though direct HTTP/curl to the
same resolve URLs works fine (observed once; see PROGRESS.md).

Beyond the plan's minimal shape assertion, this also saves a visual
before/after comparison (LR input, bicubic baseline, diffusion SR output,
and — since k=2 — a per-pixel absolute-difference map between the two
stochastic samples, previewing what Signal S1's null-space variance will
look like once Task 6 is implemented) to outputs/smoke_test_backbone/.
"""
import os

import numpy as np
import torch
from PIL import Image
from skimage import data

from src.backbone.diffusion_backbone import DEFAULT_MODEL_ID, DiffusionSRBackbone

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "smoke_test_backbone")
# Override with a local pre-downloaded snapshot dir (see checkpoints/) when
# the huggingface_hub downloader itself is the bottleneck on a given network.
MODEL_ID = os.environ.get("CHASR_MODEL_ID", DEFAULT_MODEL_ID)


def _load_real_lr_patch() -> torch.Tensor:
    """A real photo (scikit-image's bundled astronaut sample), area-downsampled
    to the backbone's required 64x64 input — a real-content LR patch rather
    than random noise, so the output is actually inspectable."""
    hr = data.astronaut()  # (512, 512, 3) uint8, real photo, no network needed
    hr_img = Image.fromarray(hr).resize((64, 64), Image.Resampling.LANCZOS)
    lr = np.asarray(hr_img).astype(np.float32) / 255.0  # (64, 64, 3) in [0, 1]
    return torch.from_numpy(lr).permute(2, 0, 1).contiguous()  # (3, 64, 64)


def _save(tensor: torch.Tensor, path: str) -> None:
    arr = tensor.detach().to(dtype=torch.float32, device="cpu").clamp(0, 1).numpy()
    arr = (arr.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lr_patch = _load_real_lr_patch()

    backbone = DiffusionSRBackbone(device="cuda", model_id=MODEL_ID)
    samples = backbone.sample_k_16x(lr_patch, k=2, base_seed=0)
    assert len(samples) == 2
    for s in samples:
        assert s.shape == (3, 1024, 1024)
    print("Output shape:", samples[0].shape)

    _save(lr_patch, os.path.join(OUTPUT_DIR, "01_lr_input_64x64.png"))

    bicubic = torch.nn.functional.interpolate(
        lr_patch.unsqueeze(0), size=(1024, 1024), mode="bicubic", align_corners=False
    ).squeeze(0).clamp(0, 1)
    _save(bicubic, os.path.join(OUTPUT_DIR, "02_bicubic_baseline_1024x1024.png"))

    _save(samples[0], os.path.join(OUTPUT_DIR, "03_diffusion_sr_sample_a_1024x1024.png"))
    _save(samples[1], os.path.join(OUTPUT_DIR, "04_diffusion_sr_sample_b_1024x1024.png"))

    diff = (samples[0].to(dtype=torch.float32, device="cpu") - samples[1].to(dtype=torch.float32, device="cpu")).abs().mean(dim=0)
    diff_normalized = (diff / (diff.max() + 1e-8)).unsqueeze(0).repeat(3, 1, 1)
    _save(diff_normalized, os.path.join(OUTPUT_DIR, "05_null_space_variance_preview.png"))

    print(f"Saved comparison images to {OUTPUT_DIR}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
