import torch

from src.fusion.infer import run_chasr
from src.fusion.model import FusionHead


class _StubBackbone:
    def sample_k_16x(self, lr_patch, k, base_seed):
        return [torch.rand(3, 1024, 1024) for _ in range(k)]

    def _upscale_4x(self, patch, seed):
        # estimate_prior_reliance_map (Task 5) calls this directly; stub it
        # so run_chasr's plumbing is testable without a real backbone.
        h, w = patch.shape[-2], patch.shape[-1]
        return torch.rand(3, h * 4, w * 4)


class _StubKernelEstimator:
    def __call__(self, lr_patch):
        return torch.tensor([[0.8, 0.8, 0.0, 0.01]])


class _StubFeatureBank:
    def knn_distance(self, query_features, k=5):
        return torch.tensor([0.5])

    def normalized_distance(self, query_features, k=5):
        return torch.tensor([0.5])


class _StubDinoEncoder:
    def __call__(self, pixel_values):
        return type("Out", (), {"pooler_output": pixel_values.mean(dim=(2, 3))})()


def test_run_chasr_returns_expected_keys_and_shapes():
    lr_patch = torch.rand(3, 64, 64)
    backbone = _StubBackbone()
    kernel_estimator = _StubKernelEstimator()
    feature_bank = _StubFeatureBank()
    dino_encoder = _StubDinoEncoder()
    fusion_head = FusionHead()

    result = run_chasr(lr_patch, backbone, kernel_estimator, feature_bank, dino_encoder, fusion_head, k=2)

    assert result["sr_image"].shape == (3, 1024, 1024)
    assert result["cause_map"].shape == (1024, 1024)
    assert result["reliability_map"].shape == (1024, 1024)
    assert result["cause_map"].dtype == torch.int64
    assert result["signal_stack"].shape == (4, 1024, 1024)
    assert result["full_signal_stack"].shape == (5, 1024, 1024)

    # Content-identity check, not just shape: S3/S4 are spatially-constant
    # (torch.full_like) for these stubs while S1/S2/luminance are not, so
    # verifying signal_stack[2]/[3] are the expected constants confirms the
    # [:4] slice actually kept S1-S4 in order rather than e.g. accidentally
    # including luminance (index 4) via an off-by-one.
    s3_channel, s4_channel = result["signal_stack"][2], result["signal_stack"][3]
    assert torch.allclose(s3_channel, torch.zeros_like(s3_channel))  # in-distribution stub kernel -> 0 mismatch
    assert torch.allclose(s4_channel, torch.full_like(s4_channel, 0.5))  # stub normalized_distance -> 0.5

    # full_signal_stack must be signal_stack's S1-S4 channels plus one more
    # (luminance) appended, not an independently-recomputed stack that could
    # silently drift from what the fusion head (and any control model
    # trained on the same 5-channel cache) actually consumes.
    assert torch.equal(result["full_signal_stack"][:4], result["signal_stack"])
