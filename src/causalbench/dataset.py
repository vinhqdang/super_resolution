"""PyTorch Dataset over xSR-CausalBench: for each source HR image, all four
controlled-injection procedures are generated deterministically per index
so the benchmark is reproducible given a fixed seed.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from src.causalbench.build_benchmark import (
    build_distribution_shift_sample,
    build_ill_posed_sample,
    build_mismatch_sample,
    build_prior_reliance_sample,
)

_PROCEDURES = ["ill_posed", "prior_reliance", "mismatch", "distribution_shift"]


class CausalBenchDataset(Dataset):
    def __init__(self, hr_images: list[np.ndarray], ood_patches: list[np.ndarray], scale: int = 16, seed: int = 0):
        self.hr_images = hr_images
        self.ood_patches = ood_patches
        self.scale = scale
        self.seed = seed

    def __len__(self) -> int:
        return len(self.hr_images) * len(_PROCEDURES)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_idx, procedure_idx = divmod(idx, len(_PROCEDURES))
        hr_source = self.hr_images[image_idx]
        procedure = _PROCEDURES[procedure_idx]
        rng = np.random.default_rng((self.seed, idx))

        if procedure == "ill_posed":
            hr, lr, label_map = build_ill_posed_sample(hr_source, rng, self.scale)
        elif procedure == "prior_reliance":
            hr, lr, label_map = build_prior_reliance_sample(hr_source, rng, self.scale)
        elif procedure == "mismatch":
            hr, lr, label_map = build_mismatch_sample(hr_source, rng, self.scale)
        else:
            # image_idx (not idx): idx is always 4*image_idx+3 for this
            # procedure, so idx % len(ood_patches) is periodic in
            # gcd(4, len(ood_patches)) and can collapse to 1-2 reachable
            # patches for any power-of-two pool size (e.g. len==4 always
            # selects ood_patches[3], never 0-2). image_idx increments by
            # 1 per source image, giving genuine round-robin coverage.
            ood_patch = self.ood_patches[image_idx % len(self.ood_patches)]
            hr, lr, label_map = build_distribution_shift_sample(hr_source, ood_patch, rng, self.scale)

        lr_t = torch.from_numpy(lr).permute(2, 0, 1).float()
        hr_t = torch.from_numpy(hr).permute(2, 0, 1).float()
        label_t = torch.from_numpy(label_map).long()
        return lr_t, hr_t, label_t
