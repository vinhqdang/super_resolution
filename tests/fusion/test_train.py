import torch

from src.causalbench.build_benchmark import CAUSE_LABELS
from src.fusion.model import FusionHead
from src.fusion.train import _cache_fingerprint, precompute_signal_stacks, train


def _synthetic_cached_items(n=4, h=8, w=8):
    items = []
    for _ in range(n):
        signal_stack = torch.rand(5, h, w)
        label_map = torch.randint(0, 5, (h, w))
        items.append((signal_stack, label_map))
    return items


def test_train_updates_reliability_head_parameters():
    # Regression test for a real bug: reliability_head had its own branch
    # off the shared trunk with no loss term of its own, so it never
    # received gradient and stayed at random init regardless of training —
    # despite being described as a "calibrated reliability score."
    cached_items = _synthetic_cached_items()
    torch.manual_seed(0)
    model = FusionHead()
    before = model.reliability_head.weight.clone()

    trained = train(cached_items, epochs=3, batch_size=2, device="cpu")

    after = trained.reliability_head.weight
    assert not torch.allclose(before, after)


def test_train_loss_includes_reliability_supervision_signal():
    # Sanity check on the supervision target itself: an all-RELIABLE label
    # map should drive the reliability head toward predicting "reliable"
    # (high) almost everywhere after a few steps, on a tiny overfit batch.
    torch.manual_seed(0)
    signal_stack = torch.rand(1, 5, 8, 8)
    label_map = torch.full((1, 8, 8), CAUSE_LABELS["RELIABLE"], dtype=torch.long)
    cached_items = [(signal_stack[0], label_map[0])]

    trained = train(cached_items, epochs=50, batch_size=1, device="cpu")
    with torch.no_grad():
        _, reliability = trained(signal_stack)
    assert reliability.mean() > 0.6


def test_cache_fingerprint_changes_with_k():
    class _FakeDataset:
        hr_images = [1, 2, 3]
        ood_patches = [1]
        scale = 16
        seed = 0

        def __len__(self):
            return 12

    fp_k2 = _cache_fingerprint(_FakeDataset(), k=2)
    fp_k4 = _cache_fingerprint(_FakeDataset(), k=4)
    assert fp_k2 != fp_k4


def test_precompute_signal_stacks_rejects_stale_cache(tmp_path):
    class _FakeDataset:
        hr_images = [1, 2]
        ood_patches = [1]
        scale = 16
        seed = 0

        def __len__(self):
            return 8

        def __getitem__(self, idx):
            return torch.rand(3, 64, 64), None, torch.zeros(1024, 1024, dtype=torch.long)

    cache_path = str(tmp_path / "cache.pt")
    torch.save({"fingerprint": {"num_hr_images": 999}, "items": []}, cache_path)

    import pytest

    with pytest.raises(ValueError, match="different .* configuration"):
        precompute_signal_stacks(_FakeDataset(), None, None, None, None, k=2, cache_path=cache_path)
