import torch

from src.fusion.model import FusionHead


def test_forward_output_shapes():
    model = FusionHead()
    signals = torch.rand(2, 5, 32, 32)
    cause_logits, reliability = model(signals)
    assert cause_logits.shape == (2, 5, 32, 32)
    assert reliability.shape == (2, 1, 32, 32)


def test_reliability_output_in_valid_range():
    model = FusionHead()
    signals = torch.rand(2, 5, 32, 32)
    _, reliability = model(signals)
    assert (reliability >= 0).all() and (reliability <= 1).all()


def test_model_can_overfit_a_tiny_synthetic_batch():
    torch.manual_seed(0)
    model = FusionHead()
    signals = torch.rand(4, 5, 16, 16)
    labels = torch.randint(0, 5, (4, 16, 16))
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(50):
        optim.zero_grad()
        cause_logits, _ = model(signals)
        loss = torch.nn.functional.cross_entropy(cause_logits, labels)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5
