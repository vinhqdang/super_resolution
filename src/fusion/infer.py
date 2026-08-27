"""Top-level CHASR entry point: LR patch in, SR image + per-pixel cause
attribution + reliability map out. Wires together Tasks 4-13. This is what
the manuscript's Experiments section (spec Section 5) calls to produce
every reported number.
"""
import torch

from src.backbone.diffusion_backbone import estimate_prior_reliance_map
from src.fusion.train import SIGNAL_STACK_K
from src.signals.degradation_mismatch import compute_degradation_mismatch
from src.signals.distribution_shift import extract_dinov2_features
from src.signals.null_space import normalized_null_space_variance


def run_chasr(lr_patch, backbone, kernel_estimator, feature_bank, dino_encoder, fusion_head, k: int = SIGNAL_STACK_K) -> dict:
    samples = backbone.sample_k_16x(lr_patch, k=k, base_seed=0)
    sr_image = samples[0]

    # normalized (not raw): matches src/fusion/train.py's build_signal_stack,
    # which is what the fusion head was actually trained against.
    s1 = normalized_null_space_variance(samples)

    s2 = estimate_prior_reliance_map(backbone, lr_patch, base_seed=0)  # see Task 5 for why this, not compute_prior_reliance

    s3_scalar = compute_degradation_mismatch(kernel_estimator, lr_patch.unsqueeze(0))
    s3 = torch.full_like(s1, s3_scalar)

    query_feats = extract_dinov2_features(dino_encoder, sr_image.unsqueeze(0))
    # normalized_distance, not raw knn_distance: matches src/fusion/train.py's
    # build_signal_stack, which is what the fusion head was actually trained
    # against — feeding raw (unbounded) distance here instead would be a
    # train/inference distribution mismatch on this input channel.
    s4_scalar = feature_bank.normalized_distance(query_feats).item()
    s4 = torch.full_like(s1, s4_scalar)

    luminance = sr_image.mean(dim=0)
    signal_stack = torch.stack([s1, s2, s3, s4, luminance], dim=0).unsqueeze(0)

    with torch.no_grad():
        cause_logits, reliability = fusion_head(signal_stack)
    cause_map = cause_logits.argmax(dim=1).squeeze(0)
    reliability_map = reliability.squeeze(0).squeeze(0)

    return {
        "sr_image": sr_image,
        "cause_map": cause_map,
        "reliability_map": reliability_map,
        "signal_stack": signal_stack.squeeze(0)[:4],  # (4, H, W): S1-S4 only, luminance (index 4) dropped — for disentanglement analysis (Task 13)
    }
