"""Fusion head (spec Section 2): the only trained component in the CHASR
pipeline. Combines the four causal signals (plus a shallow image feature)
into a calibrated per-pixel 5-way cause classification (4 causes +
"reliable") and an overall reliability score.
"""
import torch
import torch.nn as nn


class FusionHead(nn.Module):
    def __init__(self, in_channels: int = 5, num_classes: int = 5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.cause_head = nn.Conv2d(32, num_classes, 1)
        self.reliability_head = nn.Conv2d(32, 1, 1)

    def forward(self, signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(signals)
        cause_logits = self.cause_head(feats)
        reliability = torch.sigmoid(self.reliability_head(feats))
        return cause_logits, reliability
