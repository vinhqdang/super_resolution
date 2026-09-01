"""Real-ESRGAN (RRDBNet, x4plus) as a fidelity baseline for CHASR's frozen
diffusion backbone. The `realesrgan`/`basicsr` pip packages fail to build in
this environment (a known basicsr packaging bug unrelated to this project,
confirmed via `pip install basicsr` raising `KeyError: '__version__'` during
its own setup.py), so this reimplements the RRDBNet generator architecture
directly and loads the official public `RealESRGAN_x4plus.pth` weights
(downloaded via curl, same workaround pattern already used elsewhere in this
project for unreliable/broken package installs) rather than depending on
either package. Loaded with `strict=True` state-dict matching as the
correctness check: if this architecture didn't match the checkpoint exactly,
loading would raise immediately rather than silently running a wrong model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CHECKPOINT = "checkpoints/realesrgan/RealESRGAN_x4plus.pth"


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Matches RealESRGAN_x4plus.pth exactly: num_feat=64, num_block=23,
    num_grow_ch=32, scale=4 (confirmed against the checkpoint's own
    state-dict shapes before writing this, not assumed from memory alone)."""

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, num_feat: int = 64, num_block: int = 23, num_grow_ch: int = 32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # 4x via two nearest-neighbor-upsample-then-conv stages (not pixel
        # shuffle) — matches the official x4plus architecture.
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_real_esrgan_x4plus(checkpoint_path: str = DEFAULT_CHECKPOINT, device: str = "cuda") -> RRDBNet:
    model = RRDBNet()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device)


@torch.no_grad()
def upscale_16x(model: RRDBNet, lr_patch: torch.Tensor) -> torch.Tensor:
    """lr_patch: (3, 64, 64) in [0, 1]. Returns (3, 1024, 1024) in [0, 1].
    Applies the 4x model twice (64->256->1024), matching CHASR's own
    two-chained-4x-hop structure (Section 3.1 of the manuscript) for a
    same-total-factor comparison rather than a single-hop 4x-only baseline."""
    device = next(model.parameters()).device
    x = lr_patch.to(device=device, dtype=torch.float32).unsqueeze(0)
    hop1 = model(x).clamp(0, 1)
    hop2 = model(hop1).clamp(0, 1)
    return hop2.squeeze(0).cpu()
