"""Builds the DINOv2 reference feature bank from real HR training patches
(e.g. DIV2K/Flickr2K). Run manually:
conda run -n py313 python scripts/build_feature_bank.py --data-dir data/DIV2K_train_HR --out checkpoints/feature_bank.pt
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="checkpoints/feature_bank.pt")
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--max-images", type=int, default=800)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()

    paths = sorted(glob.glob(f"{args.data_dir}/*.png"))[: args.max_images]
    if not paths:
        raise FileNotFoundError(f"No .png images found under {args.data_dir}")
    features = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB").resize((args.patch_size, args.patch_size))
            tensor = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            tensor = tensor.to(device)
            out = encoder(pixel_values=tensor, interpolate_pos_encoding=True)
            features.append(out.pooler_output.cpu())

    bank = torch.cat(features, dim=0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(bank, args.out)
    print(f"Saved feature bank of shape {bank.shape} to {args.out}")


if __name__ == "__main__":
    main()
