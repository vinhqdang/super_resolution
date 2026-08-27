"""Trains the blind kernel estimator on real HR images (e.g. DIV2K train
patches). Run manually: conda run -n py313 python scripts/train_kernel_estimator.py
--data-dir data/DIV2K_train_HR --epochs 20 --out checkpoints/kernel_estimator.pt
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.signals.kernel_estimator import KernelEstimator, KernelEstimatorDataset


def load_hr_images(data_dir: str, patch_size: int = 256, max_images: int = 500) -> list[np.ndarray]:
    paths = sorted(glob.glob(f"{data_dir}/*.png"))[:max_images]
    images = []
    for p in paths:
        img = np.asarray(Image.open(p).convert("RGB").resize((patch_size, patch_size))).astype(np.float32) / 255.0
        images.append(img)
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", default="checkpoints/kernel_estimator.pt")
    args = parser.parse_args()

    hr_images = load_hr_images(args.data_dir)
    dataset = KernelEstimatorDataset(hr_images)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KernelEstimator().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for lr_batch, target_batch in loader:
            lr_batch, target_batch = lr_batch.to(device), target_batch.to(device)
            optim.zero_grad()
            pred = model(lr_batch)
            loss = torch.nn.functional.mse_loss(pred, target_batch)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"epoch {epoch}: loss {total_loss / len(loader):.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
