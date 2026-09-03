"""Capacity-matched control for the ablation (Section 5.4's own named gap:
"a jointly-fit but still simple classifier on the same four raw signals,
such as a per-pixel logistic regression, sits in the unexplored middle
ground between a single grid-searched threshold and the full convolutional
fusion head, and this pilot does not test it"). A single 1x1 convolution
applied to the same five-channel signal stack the fusion head consumes is
exactly a per-pixel logistic regression: a linear map with no spatial
context and no hidden layer, jointly fit by gradient descent like the
fusion head, but architecturally minimal rather than a 3-layer CNN. This
isolates whether the fusion head's margin over the ablation's hand-crafted
threshold baselines comes from joint fitting alone, or specifically from
its added spatial/nonlinear capacity.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class LogisticControl(nn.Module):
    def __init__(self, in_channels: int = 5, num_classes: int = 5):
        super().__init__()
        self.linear = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return self.linear(signals)


class _CachedSignalDataset(Dataset):
    def __init__(self, cached_items: list[tuple[torch.Tensor, torch.Tensor]]):
        self.cached_items = cached_items

    def __len__(self):
        return len(self.cached_items)

    def __getitem__(self, idx):
        return self.cached_items[idx]


def train_logistic_control(cached_items: list[tuple[torch.Tensor, torch.Tensor]], epochs: int = 30, batch_size: int = 4, device: str = "cuda") -> LogisticControl:
    """Same calibration-set cache, batch size, optimizer, and epoch count as
    src.fusion.train.train, so the only difference from the fusion head is
    architecture, not training budget."""
    loader = DataLoader(_CachedSignalDataset(cached_items), batch_size=batch_size, shuffle=True)
    model = LogisticControl().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _epoch in range(epochs):
        for signal_stack, label_map in loader:
            signal_stack, label_map = signal_stack.to(device), label_map.to(device)
            optim.zero_grad()
            logits = model(signal_stack)
            loss = nn.functional.cross_entropy(logits, label_map)
            loss.backward()
            optim.step()

    return model


@torch.no_grad()
def predict_logistic_control(model: LogisticControl, signal_stack: torch.Tensor) -> torch.Tensor:
    """signal_stack: (C, H, W), single item with no batch dimension. Returns
    (H, W) of predicted class indices."""
    device = next(model.parameters()).device
    logits = model(signal_stack.unsqueeze(0).to(device))
    return logits.argmax(dim=1).squeeze(0)
