"""Frozen extreme-SR backbone B: two chained 4x diffusion super-resolution
hops (64x64 -> 256x256 -> 1024x1024, i.e. 16x total), matching the
Chain-of-Zoom autoregressive-zoom idea (arXiv:2505.18600) but built on the
public, directly downloadable stabilityai/stable-diffusion-x4-upscaler
checkpoint for reproducibility on a single 8GB GPU. Each hop is DDNM-
projected (src.backbone.ddnm_projection) against its true observation so
every returned sample is exactly degradation-consistent — required for the
Signal 1 (null-space variance) computation in src/signals/null_space.py.

The second hop is applied per-tile (2x2 tiling of the 256x256 intermediate
into four 128x128 tiles, each upscaled to 512x512 and stitched back to
1024x1024) to stay within the 8GB VRAM budget — see Global Constraints.
"""
import torch

from src.backbone.ddnm_projection import ddnm_project


DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"


class DiffusionSRBackbone:
    def __init__(self, pipeline=None, device: str = "cuda", model_id: str = DEFAULT_MODEL_ID):
        """model_id: a hub repo id (default) or a local directory containing
        a pre-downloaded snapshot in the same layout — useful when the
        huggingface_hub downloader itself is the bottleneck (observed: HF's
        Xet transfer and its plain-HTTP fallback both stalled indefinitely
        on one network here, while `curl` to the same resolve URLs ran at
        several MB/s) and the checkpoint was fetched by hand instead."""
        self.device = device
        self.model_id = model_id
        self._pipeline = pipeline  # lazy-loaded below if None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from diffusers import StableDiffusionUpscalePipeline

            self._pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                self.model_id, torch_dtype=torch.float16, variant="fp16"
            ).to(self.device)
            self._pipeline.enable_attention_slicing()
        return self._pipeline

    def _upscale_4x(self, patch: torch.Tensor, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        # StableDiffusionUpscalePipeline requires `prompt` or `prompt_embeds` to
        # be set (raises ValueError otherwise) — pass an empty prompt so
        # upscaling stays domain-agnostic/text-free, matching the blind xSR
        # framing of the backbone (no semantic guidance injected).
        # image must be explicitly batched (1, C, H, W): passing an
        # unbatched (C, H, W) tensor makes check_inputs read the channel
        # dim as a batch of 3 single-channel images and raise a batch-size
        # mismatch against the (batch-size-1) prompt.
        result = self.pipeline(image=patch.unsqueeze(0), prompt="", generator=generator, num_inference_steps=20, output_type="pt")
        return result.images[0]

    def sample_k_16x(self, lr_patch: torch.Tensor, k: int, base_seed: int) -> list[torch.Tensor]:
        """lr_patch: (3, 64, 64). Returns k tensors of shape (3, 1024, 1024)."""
        samples = []
        for i in range(k):
            seed = base_seed + i
            hop1 = self._upscale_4x(lr_patch, seed=seed)  # (3, 256, 256)
            # ddnm_project does plain tensor arithmetic with no device/dtype
            # casting of its own — the pipeline output (e.g. CUDA fp16) and
            # the caller-supplied lr_patch (e.g. CPU fp32) must match, so
            # match the reference to whatever the pipeline actually returned.
            lr_ref = lr_patch.to(device=hop1.device, dtype=hop1.dtype)
            hop1 = ddnm_project(hop1, lr_ref, scale=4)

            tiles = []
            for ty in range(2):
                for tx in range(2):
                    tile_lr = hop1[:, ty * 128 : (ty + 1) * 128, tx * 128 : (tx + 1) * 128]
                    tile_hr = self._upscale_4x(tile_lr, seed=seed + 1000 + ty * 2 + tx)  # (3, 512, 512)
                    tile_lr_ref = tile_lr.to(device=tile_hr.device, dtype=tile_hr.dtype)
                    tile_hr = ddnm_project(tile_hr, tile_lr_ref, scale=4)
                    tiles.append((ty, tx, tile_hr))

            stitched = torch.zeros(3, 1024, 1024, device=tiles[0][2].device, dtype=tiles[0][2].dtype)
            for ty, tx, tile_hr in tiles:
                stitched[:, ty * 512 : (ty + 1) * 512, tx * 512 : (tx + 1) * 512] = tile_hr

            lr_ref_final = lr_patch.to(device=stitched.device, dtype=stitched.dtype)
            final = ddnm_project(stitched, lr_ref_final, scale=16)
            samples.append(final)
        return samples


def estimate_prior_reliance_map(backbone: DiffusionSRBackbone, lr_patch: torch.Tensor, base_seed: int = 0, eps: float = 1e-2) -> torch.Tensor:
    """Signal S2 (spec Section 2), real-pipeline implementation. Generic
    finite-difference sensitivity (src/signals/prior_reliance.py) assumes a
    continuously-perturbable forward_fn and latent code; this backbone's
    stochasticity is a discrete integer seed, not a continuous latent, so
    S2 is estimated directly here instead: sensitivity of the raw (pre-
    DDNM-projection) hop-1 output to a small perturbation of the LR
    evidence at fixed seed (g_evidence) vs. to a different seed at fixed LR
    evidence (g_prior, a discrete perturbation). Both are left as raw
    (un-normalized-by-eps) output-magnitude changes rather than true
    derivatives: g_prior has no natural step size to divide by (it's a full
    reseed, not an infinitesimal step), and dividing only g_evidence by a
    small eps would blow its scale up relative to g_prior and bias S2
    toward 0 regardless of how the model actually behaves. Comparing raw
    magnitudes instead answers the intended question directly — did the
    output change more from a small nudge to real evidence, or from
    changing nothing but the random seed — on the same units.
    Computed once at hop-1 resolution (256x256) — only 2 extra single-hop
    diffusion calls, not full 16x chains — then nearest-upsampled to
    1024x1024 to align with S1/S3/S4.
    """
    base_hop1 = backbone._upscale_4x(lr_patch, seed=base_seed)  # (3, 256, 256)

    lr_perturbed = lr_patch + eps * torch.randn_like(lr_patch)
    hop1_lr_perturbed = backbone._upscale_4x(lr_perturbed, seed=base_seed)
    lr_perturbed_ref = hop1_lr_perturbed.to(device=base_hop1.device, dtype=base_hop1.dtype)
    g_evidence = (lr_perturbed_ref - base_hop1).abs().mean(dim=0)  # (256, 256)

    hop1_seed_perturbed = backbone._upscale_4x(lr_patch, seed=base_seed + 1)
    seed_perturbed_ref = hop1_seed_perturbed.to(device=base_hop1.device, dtype=base_hop1.dtype)
    g_prior = (seed_perturbed_ref - base_hop1).abs().mean(dim=0)  # (256, 256)

    s2_hop1 = g_prior / (g_prior + g_evidence + 1e-8)
    return torch.nn.functional.interpolate(
        s2_hop1.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0), size=(1024, 1024), mode="nearest"
    ).squeeze(0).squeeze(0)
