import importlib

REQUIRED_MODULES = [
    "torch", "diffusers", "transformers", "lpips",
    "skimage", "cv2", "scipy", "sklearn", "easyocr", "numpy", "PIL",
]


def test_required_modules_importable():
    missing = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    assert not missing, f"Missing modules: {missing}"


def test_cuda_available():
    import torch
    assert torch.cuda.is_available(), "CUDA GPU not detected — required for backbone/signal tasks"
