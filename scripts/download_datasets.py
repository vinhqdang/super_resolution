"""Downloads/documents the source datasets used to build xSR-CausalBench
and for backbone/estimator training and evaluation. DIV2K has a stable
public direct-download URL; the other sets require manual steps due to
registration/access-agreement requirements — verify URLs are still live
before relying on them, hosting can change.

Run: conda run -n py313 python scripts/download_datasets.py --out-dir data/
"""
import argparse
import os
import zipfile

import requests

DIV2K_URLS = {
    "DIV2K_train_HR": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "DIV2K_valid_HR": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}

MANUAL_DATASETS = {
    "Urban100": "https://github.com/jbhuang0604/SelfExSR (request access per repo instructions)",
    "Manga109": "http://www.manga109.org/en/ (requires registration + usage agreement)",
    "RealSR": "https://github.com/csjcai/RealSR (request access per repo instructions)",
    "DRealSR": "https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution (request access per repo instructions)",
}


def _safe_extract(zf: zipfile.ZipFile, out_dir: str) -> None:
    """Rejects any archive member whose resolved path would land outside
    out_dir (a "Zip Slip" path-traversal entry, e.g. "../../evil") before
    extracting anything — zipfile.extractall() alone does not validate this."""
    out_dir_abs = os.path.abspath(out_dir)
    for member in zf.namelist():
        resolved = os.path.abspath(os.path.join(out_dir, member))
        if not (resolved == out_dir_abs or resolved.startswith(out_dir_abs + os.sep)):
            raise ValueError(f"Refusing to extract unsafe archive member: {member}")
    zf.extractall(out_dir)


def download_and_extract(name: str, url: str, out_dir: str):
    zip_path = os.path.join(out_dir, f"{name}.zip")
    print(f"Downloading {name} from {url} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    with zipfile.ZipFile(zip_path) as zf:
        _safe_extract(zf, out_dir)
    os.remove(zip_path)
    print(f"Extracted {name} to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for name, url in DIV2K_URLS.items():
        try:
            download_and_extract(name, url, args.out_dir)
        except Exception as e:
            print(f"Failed to auto-download {name}: {e}. Download manually from {url}")

    print("\nThe following datasets require manual download (registration/access agreement):")
    for name, info in MANUAL_DATASETS.items():
        print(f"  {name}: {info}")


if __name__ == "__main__":
    main()
