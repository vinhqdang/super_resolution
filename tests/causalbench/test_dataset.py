import numpy as np
import torch

from src.causalbench.dataset import CausalBenchDataset


def test_dataset_getitem_shapes_and_dtypes(tmp_path):
    rng = np.random.default_rng(0)
    hr_images = [rng.uniform(0, 1, size=(256, 256, 3)).astype(np.float32) for _ in range(4)]
    ood_patches = [rng.uniform(0, 1, size=(64, 64, 3)).astype(np.float32) for _ in range(2)]

    dataset = CausalBenchDataset(hr_images, ood_patches, scale=16, seed=0)
    assert len(dataset) == len(hr_images) * 4  # 4 procedures per source image

    lr, hr, label_map = dataset[0]
    assert isinstance(lr, torch.Tensor) and lr.shape[0] == 3
    assert isinstance(hr, torch.Tensor) and hr.shape[0] == 3
    assert isinstance(label_map, torch.Tensor) and label_map.dtype == torch.int64
    assert label_map.shape == hr.shape[1:]


def test_dataset_covers_all_four_procedures():
    rng = np.random.default_rng(1)
    hr_images = [rng.uniform(0, 1, size=(256, 256, 3)).astype(np.float32) for _ in range(1)]
    ood_patches = [rng.uniform(0, 1, size=(64, 64, 3)).astype(np.float32)]

    dataset = CausalBenchDataset(hr_images, ood_patches, scale=16, seed=0)
    seen_labels = set()
    for i in range(len(dataset)):
        _, _, label_map = dataset[i]
        seen_labels.update(torch.unique(label_map).tolist())
    assert seen_labels == {0, 1, 2, 3, 4}


def test_dataset_distribution_shift_uses_more_than_one_ood_patch():
    # Regression test: idx % len(ood_patches) (idx always 4*image_idx+3 for
    # this procedure) collapses to 1-2 reachable patches for any
    # power-of-two ood_patches pool — with len==4 (this test), idx % 4 was
    # always 3, so ood_patches[3] was the ONLY patch ever selected across
    # every source image. Use 4 visually-distinct (distinct constant-value)
    # patches so which one was actually blended in is directly observable.
    size = 256
    hr_images = [np.random.default_rng(i).uniform(0, 1, size=(size, size, 3)).astype(np.float32) for i in range(4)]
    ood_patches = [np.full((64, 64, 3), v, dtype=np.float32) for v in (0.0, 0.25, 0.5, 0.75)]

    dataset = CausalBenchDataset(hr_images, ood_patches, scale=16, seed=0)
    used_values = set()
    for image_idx in range(4):
        idx = image_idx * 4 + 3  # distribution_shift is always procedure_idx 3
        _, hr, label_map = dataset[idx]
        region_mask = label_map == 3  # CAUSE_LABELS["DISTRIBUTION_SHIFT"]
        ys, xs = torch.where(region_mask)
        used_values.add(round(hr[0, ys[0], xs[0]].item(), 2))
    assert len(used_values) > 1
