"""Signal disentanglement analysis (spec Section 3 risk / Section 5, item
5): correlation matrix among S1-S4 must be reported honestly rather than
assuming clean separation — see spec Section 7 open risks.
"""
import torch


def signal_correlation_matrix(signals: torch.Tensor) -> torch.Tensor:
    """signals: (4, N) flattened per-signal samples. Returns (4, 4) Pearson
    correlation matrix."""
    if signals.shape[1] < 2:
        raise ValueError("Need at least 2 samples to compute correlation")
    centered = signals - signals.mean(dim=1, keepdim=True)
    cov = centered @ centered.T / (signals.shape[1] - 1)
    std = torch.sqrt(torch.diag(cov))
    denom = std.unsqueeze(0) * std.unsqueeze(1)
    return cov / denom
