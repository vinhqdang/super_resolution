import torch

from src.backbone.diffusion_backbone import DiffusionSRBackbone


class _StubPipeline:
    """Returns a deterministic-shape random image; ignores the prompt/seed logic
    beyond producing a torch generator-seeded tensor, standing in for the real
    diffusers pipeline so this test needs no network/GPU download."""

    def __call__(self, image, prompt="", generator=None, num_inference_steps=20, output_type="pt"):
        # Interface-drift guard: the real StableDiffusionUpscalePipeline raises
        # ValueError if both `prompt` and `prompt_embeds` are None — catch a
        # caller regression here instead of only at real-model runtime.
        assert prompt is not None, "caller must pass a non-None prompt"
        # image: PIL.Image or tensor sized (h, w); we only need the output 4x bigger.
        h, w = image.shape[-2], image.shape[-1]
        g = generator
        out = torch.rand(1, 3, h * 4, w * 4, generator=g)
        return type("Result", (), {"images": out})()


def test_sample_k_16x_output_shape_and_count():
    backbone = DiffusionSRBackbone(pipeline=_StubPipeline(), device="cpu")
    lr_patch = torch.rand(3, 64, 64)
    samples = backbone.sample_k_16x(lr_patch, k=2, base_seed=0)
    assert len(samples) == 2
    for s in samples:
        assert s.shape == (3, 1024, 1024)


def test_sample_k_16x_is_degradation_consistent_at_each_hop():
    def _average_pool(x, scale):
        c, h, w = x.shape
        return x.view(c, h // scale, scale, w // scale, scale).mean(dim=(2, 4))

    backbone = DiffusionSRBackbone(pipeline=_StubPipeline(), device="cpu")
    lr_patch = torch.rand(3, 64, 64)
    samples = backbone.sample_k_16x(lr_patch, k=1, base_seed=0)
    final = samples[0]
    reprojected_full = _average_pool(final, scale=16)
    assert torch.allclose(reprojected_full, lr_patch, atol=1e-4)


def test_sample_k_16x_different_seeds_give_different_samples():
    backbone = DiffusionSRBackbone(pipeline=_StubPipeline(), device="cpu")
    lr_patch = torch.rand(3, 64, 64)
    samples = backbone.sample_k_16x(lr_patch, k=2, base_seed=0)
    assert not torch.allclose(samples[0], samples[1])


def test_sample_k_16x_zero_samples_returns_empty_list():
    backbone = DiffusionSRBackbone(pipeline=_StubPipeline(), device="cpu")
    lr_patch = torch.rand(3, 64, 64)
    assert backbone.sample_k_16x(lr_patch, k=0, base_seed=0) == []


class _RecordingStubPipeline:
    """Wraps _StubPipeline but records every (image_shape, prompt) call so
    tests can verify DDNM-projection-relevant intermediate resolutions
    (hop1 and each tile) are actually produced, not just the final stitch."""

    def __init__(self):
        self._inner = _StubPipeline()
        self.calls = []

    def __call__(self, image, prompt="", generator=None, num_inference_steps=20, output_type="pt"):
        self.calls.append(tuple(image.shape))
        return self._inner(image, prompt=prompt, generator=generator, num_inference_steps=num_inference_steps, output_type=output_type)


def test_sample_k_16x_calls_pipeline_at_hop1_and_each_tile_resolution():
    pipeline = _RecordingStubPipeline()
    backbone = DiffusionSRBackbone(pipeline=pipeline, device="cpu")
    lr_patch = torch.rand(3, 64, 64)
    backbone.sample_k_16x(lr_patch, k=1, base_seed=0)
    # 1 hop1 call at (1, 3, 64, 64) batched input + 4 tile calls at (1, 3, 128, 128).
    assert pipeline.calls.count((1, 3, 64, 64)) == 1
    assert pipeline.calls.count((1, 3, 128, 128)) == 4
    assert len(pipeline.calls) == 5


def test_sample_k_16x_applies_ddnm_projection_at_every_stage_not_just_final():
    """Spies on ddnm_project to confirm it's invoked once for hop1, once per
    tile, and once for the final stitch (6 calls for k=1) — not just at the
    end, which the shape/consistency tests alone wouldn't distinguish since
    average-pool composition is associative."""
    import unittest.mock as mock

    import src.backbone.diffusion_backbone as backbone_module

    backbone = DiffusionSRBackbone(pipeline=_StubPipeline(), device="cpu")
    lr_patch = torch.rand(3, 64, 64)

    with mock.patch.object(backbone_module, "ddnm_project", wraps=backbone_module.ddnm_project) as spy:
        backbone.sample_k_16x(lr_patch, k=1, base_seed=0)

    scales_used = [call.kwargs["scale"] for call in spy.call_args_list]
    assert spy.call_count == 6  # 1 hop1 + 4 tiles + 1 final
    assert scales_used.count(4) == 5  # hop1 + 4 tiles
    assert scales_used.count(16) == 1  # final stitch


class _ContentOnlyPipeline:
    """Output depends only on the input image (nearest-neighbor 4x upsample)
    and completely ignores the generator/seed — the opposite extreme from
    _StubPipeline. Used to verify estimate_prior_reliance_map reports near-
    zero prior reliance when the seed provably doesn't matter."""

    def __call__(self, image, prompt="", generator=None, num_inference_steps=20, output_type="pt"):
        out = torch.nn.functional.interpolate(image, scale_factor=4, mode="nearest")
        return type("Result", (), {"images": out})()


def test_estimate_prior_reliance_map_output_shape():
    from src.backbone.diffusion_backbone import estimate_prior_reliance_map

    backbone = DiffusionSRBackbone(pipeline=_StubPipeline(), device="cpu")
    lr_patch = torch.rand(3, 64, 64)
    s2 = estimate_prior_reliance_map(backbone, lr_patch)
    assert s2.shape == (1024, 1024)
    assert (s2 >= 0).all() and (s2 <= 1).all()


def test_estimate_prior_reliance_map_high_when_only_seed_matters():
    from src.backbone.diffusion_backbone import estimate_prior_reliance_map

    backbone = DiffusionSRBackbone(pipeline=_StubPipeline(), device="cpu")
    lr_patch = torch.rand(3, 64, 64)
    s2 = estimate_prior_reliance_map(backbone, lr_patch)
    assert s2.mean() > 0.9


def test_estimate_prior_reliance_map_low_when_only_content_matters():
    from src.backbone.diffusion_backbone import estimate_prior_reliance_map

    backbone = DiffusionSRBackbone(pipeline=_ContentOnlyPipeline(), device="cpu")
    lr_patch = torch.rand(3, 64, 64)
    s2 = estimate_prior_reliance_map(backbone, lr_patch)
    assert s2.mean() < 0.1


class _MixedSensitivityPipeline:
    """Reacts to BOTH content and seed, at a weighting chosen so their raw
    (un-normalized) output-magnitude contributions land at comparable scale
    given the default eps=1e-2 perturbation — the regime the two degenerate
    stub pipelines above (_StubPipeline, _ContentOnlyPipeline) don't
    exercise. Guards against the S2 formula silently reverting to an
    eps-scaled g_evidence that would bias every real (non-degenerate)
    pipeline toward S2 ~= 0."""

    def __init__(self, content_weight: float = 0.976):
        self.content_weight = content_weight

    def __call__(self, image, prompt="", generator=None, num_inference_steps=20, output_type="pt"):
        h, w = image.shape[-2], image.shape[-1]
        content_term = torch.nn.functional.interpolate(image, scale_factor=4, mode="nearest")
        prior_term = torch.rand(1, 3, h * 4, w * 4, generator=generator)
        out = self.content_weight * content_term + (1 - self.content_weight) * prior_term
        return type("Result", (), {"images": out})()


def test_estimate_prior_reliance_map_is_mid_range_under_comparable_sensitivity():
    from src.backbone.diffusion_backbone import estimate_prior_reliance_map

    backbone = DiffusionSRBackbone(pipeline=_MixedSensitivityPipeline(), device="cpu")
    lr_patch = torch.rand(3, 64, 64)
    s2 = estimate_prior_reliance_map(backbone, lr_patch)
    assert 0.3 < s2.mean() < 0.7
