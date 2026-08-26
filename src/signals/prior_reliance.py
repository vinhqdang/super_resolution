"""Signal S2 (spec Section 2): local sensitivity of the output to LR input
evidence vs. to the prior's latent/noise code, via finite-difference
perturbation so it works against any black-box forward_fn (the real
diffusion backbone is not cleanly differentiable end-to-end through
sampling steps).

Known limitation: at a pixel where forward_fn is locally insensitive to
BOTH lr and z (g_evidence == g_prior == 0, e.g. a constant or saturated
output), the ratio evaluates to 0 — read as "fully evidence-driven" —
rather than the more accurate "indeterminate: neither factor explains
this output." The 1e-8 denominator guard only prevents a NaN/Inf crash;
it does not distinguish this degenerate case from genuine low prior
reliance. Callers relying on this as ground truth should treat a near-zero
(g_prior + g_evidence) as a separate "undefined" case if that distinction
matters for their use.
"""
import torch


def compute_prior_reliance(forward_fn, lr: torch.Tensor, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    base = forward_fn(lr, z)  # (C, H, W)

    lr_perturbed = lr + eps * torch.randn_like(lr)
    out_lr_perturbed = forward_fn(lr_perturbed, z)
    g_evidence = (out_lr_perturbed - base).abs().mean(dim=0) / eps  # (H, W)

    z_perturbed = z + eps * torch.randn_like(z)
    out_z_perturbed = forward_fn(lr, z_perturbed)
    g_prior = (out_z_perturbed - base).abs().mean(dim=0) / eps  # (H, W)

    return g_prior / (g_prior + g_evidence + 1e-8)
