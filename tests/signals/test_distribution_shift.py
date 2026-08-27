import torch

from src.signals.distribution_shift import FeatureBank, extract_dinov2_features


class _StubEncoder:
    """Maps each patch to its own mean-pooled RGB value as a fake 3-D
    'feature' so distance-to-bank behavior is easy to reason about without
    downloading DINOv2."""

    def __call__(self, pixel_values):
        feats = pixel_values.mean(dim=(2, 3))  # (B, 3)
        return type("Out", (), {"pooler_output": feats})()


def test_extract_dinov2_features_shape():
    encoder = _StubEncoder()
    patches = torch.rand(5, 3, 32, 32)
    feats = extract_dinov2_features(encoder, patches)
    assert feats.shape == (5, 3)


def test_in_distribution_query_has_low_knn_distance():
    # k=1 (not the plan's original k=5): with only 100 reference points in
    # 8 dimensions, curse-of-dimensionality means even a near-duplicate
    # query's 2nd-5th nearest neighbors are "generic" random points at
    # typical inter-point distance (~0.5 here) — averaging k=5 washes out
    # the one genuinely-close match. k=1 isolates the property this test
    # actually checks: a near-duplicate of a banked point resolves to ~0.
    reference = torch.rand(100, 8)
    bank = FeatureBank(reference)
    in_dist_query = reference[:3] + 0.001 * torch.randn(3, 8)
    dist = bank.knn_distance(in_dist_query, k=1)
    assert dist.shape == (3,)
    assert dist.mean() < 0.1


def test_ood_query_has_high_knn_distance():
    reference = torch.rand(100, 8)  # roughly in [0, 1]
    bank = FeatureBank(reference)
    ood_query = torch.full((3, 8), 10.0)  # far outside [0, 1]
    dist = bank.knn_distance(ood_query, k=5)
    assert dist.mean() > 5.0


def test_default_k_still_separates_in_distribution_from_ood_relatively():
    # Restores coverage of the actual production default (k=5) dropped by
    # switching the near-duplicate test above to k=1: at k=5 the absolute
    # in-distribution distance is washed out by curse-of-dimensionality
    # (see module docstring), but it should still be well below a
    # genuinely far-OOD query's distance — a relative comparison, not an
    # absolute threshold, is the property that actually holds at k=5.
    reference = torch.rand(100, 8)
    bank = FeatureBank(reference)
    in_dist_query = reference[:3] + 0.001 * torch.randn(3, 8)
    ood_query = torch.full((3, 8), 10.0)
    in_dist = bank.knn_distance(in_dist_query, k=5).mean()
    ood_dist = bank.knn_distance(ood_query, k=5).mean()
    assert in_dist < ood_dist


class _ConfigLikeStubEncoder:
    """Has a `.config` attribute (like a real transformers model) but does
    NOT actually accept interpolate_pos_encoding — the case the old
    hasattr(encoder, "config") heuristic would have silently misclassified
    and called with the wrong signature."""

    config = object()

    def __call__(self, pixel_values):
        feats = pixel_values.mean(dim=(2, 3))
        return type("Out", (), {"pooler_output": feats})()


def test_extract_dinov2_features_falls_back_for_config_like_stub_without_kwarg_support():
    encoder = _ConfigLikeStubEncoder()
    patches = torch.rand(5, 3, 32, 32)
    feats = extract_dinov2_features(encoder, patches)
    assert feats.shape == (5, 3)


def test_normalized_distance_is_bounded_and_separates_in_distribution_from_ood():
    reference = torch.rand(100, 8)
    bank = FeatureBank(reference)
    in_dist_query = reference[:3] + 0.001 * torch.randn(3, 8)
    ood_query = torch.full((3, 8), 10.0)

    in_dist_score = bank.normalized_distance(in_dist_query, k=5)
    ood_score = bank.normalized_distance(ood_query, k=5)

    assert (in_dist_score >= 0).all() and (in_dist_score <= 1).all()
    assert (ood_score >= 0).all() and (ood_score <= 1).all()
    assert in_dist_score.mean() < ood_score.mean()
