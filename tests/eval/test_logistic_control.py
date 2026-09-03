import torch

from src.eval.logistic_control import LogisticControl, predict_logistic_control, train_logistic_control


def _synthetic_cached_items(n=4, h=8, w=8):
    items = []
    for _ in range(n):
        signal_stack = torch.rand(5, h, w)
        label_map = torch.randint(0, 5, (h, w))
        items.append((signal_stack, label_map))
    return items


def test_forward_shape_matches_input_spatial_dims():
    model = LogisticControl()
    signals = torch.rand(2, 5, 8, 8)
    logits = model(signals)
    assert logits.shape == (2, 5, 8, 8)


def test_linear_layer_has_no_spatial_context():
    # A 1x1 conv with no padding beyond kernel_size=1 has a receptive field
    # of exactly one pixel — this is the property that makes LogisticControl
    # a genuine per-pixel logistic regression rather than a hidden CNN.
    model = LogisticControl()
    assert model.linear.kernel_size == (1, 1)
    assert isinstance(model.linear, torch.nn.Conv2d)
    # Exactly one linear layer, no nonlinearity, no hidden layer.
    assert len(list(model.parameters())) == 2  # weight, bias


def test_train_updates_parameters():
    cached_items = _synthetic_cached_items()
    torch.manual_seed(0)
    model = LogisticControl()
    before = model.linear.weight.clone()

    trained = train_logistic_control(cached_items, epochs=5, batch_size=2, device="cpu")

    after = trained.linear.weight
    assert not torch.allclose(before, after)


def test_train_overfits_tiny_single_class_batch():
    torch.manual_seed(0)
    signal_stack = torch.rand(5, 8, 8)
    label_map = torch.full((8, 8), 2, dtype=torch.long)
    cached_items = [(signal_stack, label_map)]

    trained = train_logistic_control(cached_items, epochs=200, batch_size=1, device="cpu")
    predicted = predict_logistic_control(trained, signal_stack)

    assert (predicted == 2).float().mean() > 0.9


def test_predict_returns_spatial_shape_without_batch_dim():
    torch.manual_seed(0)
    model = LogisticControl()
    signal_stack = torch.rand(5, 8, 8)
    predicted = predict_logistic_control(model, signal_stack)
    assert predicted.shape == (8, 8)
    assert predicted.dtype in (torch.int64, torch.long)
