"""Signal S4 (spec Section 2): distance-to-training-distribution in a
frozen DINOv2 feature space. `encoder` is injected so tests avoid
downloading the real facebook/dinov2-small checkpoint — production callers
pass transformers.AutoModel.from_pretrained("facebook/dinov2-small").

Known limitation — curse of dimensionality at production scale: k-NN
distance averaged over k>1 neighbors washes out separation between
"near-duplicate of a training point" and "generic in-range point" once
the reference bank is sparse relative to the feature dimensionality (e.g.
~800 reference images in DINOv2's ~384-768 dim feature space). A
near-duplicate query's own single nearest neighbor is close to 0, but its
2nd-5th nearest neighbors are typical "generic" reference points at a
much larger, roughly constant distance — so the k=5 average barely
distinguishes a near-duplicate from a merely-plausible in-range query,
while still clearly separating genuinely far-OOD queries (order-of-
magnitude larger distance). Confirmed on a synthetic proxy at this scale
before relying on the real feature bank; no empirical validation of the
real facebook/dinov2-small bank's actual separation exists yet — do that
before trusting S4 as a fine-grained (rather than only coarse) signal.
`normalized_distance` below rescales the raw distance into [0,1] but does
NOT fix this underlying separation loss.
"""
import torch
from sklearn.neighbors import NearestNeighbors


def extract_dinov2_features(encoder, patches: torch.Tensor) -> torch.Tensor:
    """patches: (B, 3, H, W). Returns (B, D). interpolate_pos_encoding=True is
    required for the real model because CHASR patches (e.g. 1024x1024 SR
    output) are not the fixed size DINOv2 was pretrained at; omitting it
    raises a position-embedding size mismatch at inference time.

    Tries the real transformers calling convention (pixel_values= kwarg
    plus interpolate_pos_encoding=True) first and falls back to a plain
    positional call on TypeError, rather than duck-typing on an unrelated
    attribute (e.g. hasattr(encoder, "config")) — a heuristic that would
    silently misclassify any stub/mock that happens to expose a `.config`
    attribute without actually accepting these kwargs.
    """
    try:
        out = encoder(pixel_values=patches, interpolate_pos_encoding=True)
    except TypeError:
        out = encoder(patches)
    return out.pooler_output


class FeatureBank:
    def __init__(self, reference_features: torch.Tensor):
        self.reference = reference_features.detach().cpu().numpy()
        self._nn = NearestNeighbors()
        self._nn.fit(self.reference)
        self._self_scale_cache: dict[int, float] = {}

    def knn_distance(self, query_features: torch.Tensor, k: int = 5) -> torch.Tensor:
        query_np = query_features.detach().cpu().numpy()
        k = min(k, len(self.reference))
        distances, _ = self._nn.kneighbors(query_np, n_neighbors=k)
        return torch.from_numpy(distances.mean(axis=1)).float()

    def _self_consistency_scale(self, k: int) -> float:
        """Typical (mean) k-NN distance among the reference bank's own
        points to each other (self-matches excluded by sklearn when no
        query is given) — used to rescale knn_distance into a self-
        calibrated score without assuming any fixed feature-space
        magnitude a priori (DINOv2 feature norms aren't known ahead of
        building the real bank). Cached per k since it's static for a
        given bank."""
        if k not in self._self_scale_cache:
            k_eff = min(k, len(self.reference) - 1)
            distances, _ = self._nn.kneighbors(n_neighbors=k_eff)
            self._self_scale_cache[k] = float(distances.mean())
        return self._self_scale_cache[k]

    def normalized_distance(self, query_features: torch.Tensor, k: int = 5) -> torch.Tensor:
        """Self-calibrated [0, 1] version of knn_distance:
        distance / (distance + scale), where scale is the reference
        bank's own typical internal k-NN distance. ~0.5 at "typical
        in-bank" distance, -> 1 for far OOD, -> 0 for a near-duplicate.
        Use this (not raw knn_distance) when feeding S4 into the fusion
        head, since the head's other input channels (S2, S3, luminance)
        are already bounded to [0, 1] — feeding S4's raw, unbounded,
        arbitrary-scale distance alongside them risks it dominating or
        being invisible to early conv layers depending on its incidental
        magnitude. See module docstring: this rescaling does not fix the
        k=5 separation-washout limitation, only the scale mismatch.
        """
        raw = self.knn_distance(query_features, k=k)
        scale = self._self_consistency_scale(k)
        return raw / (raw + scale + 1e-8)
