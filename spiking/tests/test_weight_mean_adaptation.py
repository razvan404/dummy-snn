import torch

from spiking.threshold import WeightMeanAdaptation


def _make(**kw) -> WeightMeanAdaptation:
    defaults = dict(min_threshold=1.0, learning_rate=0.1, target_mean=0.5)
    defaults.update(kw)
    return WeightMeanAdaptation(**defaults)


def test_below_target_raises_threshold():
    """mean(W) < target ⇒ delta > 0 ⇒ threshold ↑."""
    rule = _make()
    weights = torch.full((4, 25), 0.3)  # mean = 0.3, target = 0.5
    th = torch.full((4,), 5.0)
    new = rule.update(th, spike_times=torch.zeros(4), weights=weights)
    assert torch.allclose(new, torch.full_like(th, 5.0 + 0.1 * 0.2))


def test_above_target_lowers_threshold():
    """mean(W) > target ⇒ delta < 0 ⇒ threshold ↓."""
    rule = _make()
    weights = torch.full((4, 25), 0.7)
    th = torch.full((4,), 5.0)
    new = rule.update(th, spike_times=torch.zeros(4), weights=weights)
    assert torch.allclose(new, torch.full_like(th, 5.0 - 0.1 * 0.2))


def test_min_threshold_clamp():
    """Strong push below min_threshold is clamped."""
    rule = _make(min_threshold=2.0, learning_rate=10.0)
    weights = torch.full((1, 4), 0.9)  # delta = 10*(0.5-0.9) = -4
    th = torch.tensor([3.0])
    new = rule.update(th, spike_times=torch.zeros(1), weights=weights)
    assert torch.allclose(new, torch.tensor([2.0]))


def test_per_neuron_independent():
    """Different neurons get different deltas based on their own mean."""
    rule = _make()
    weights = torch.stack([torch.full((25,), 0.2), torch.full((25,), 0.8)])
    th = torch.full((2,), 5.0)
    new = rule.update(th, spike_times=torch.zeros(2), weights=weights)
    assert torch.allclose(new, torch.tensor([5.0 + 0.03, 5.0 - 0.03]))


def test_supports_4d_weights():
    """conv layers may pass (N, C, kH, kW) — flatten before mean."""
    rule = _make()
    weights = torch.full((3, 2, 5, 5), 0.4)
    th = torch.full((3,), 5.0)
    new = rule.update(th, spike_times=torch.zeros(3), weights=weights)
    assert torch.allclose(new, torch.full_like(th, 5.0 + 0.1 * 0.1))


def test_use_winner_applies_uniform_delta():
    """In winner mode, all thresholds shift by the same amount based on winner's mean."""
    rule = _make(use_winner=True)
    weights = torch.stack([torch.full((25,), 0.2), torch.full((25,), 0.9)])
    th = torch.tensor([4.0, 6.0])
    new = rule.update(
        th,
        spike_times=torch.zeros(2),
        weights=weights,
        neurons_to_learn=torch.tensor([0]),
    )
    expected_delta = 0.1 * (0.5 - 0.2)  # winner is neuron 0
    assert torch.allclose(new, th + expected_delta)


def test_use_winner_no_winners_passthrough():
    """In winner mode, missing winners ⇒ thresholds unchanged."""
    rule = _make(use_winner=True)
    th = torch.tensor([4.0, 6.0])
    new = rule.update(
        th,
        spike_times=torch.zeros(2),
        weights=torch.full((2, 25), 0.3),
        neurons_to_learn=torch.tensor([], dtype=torch.long),
    )
    assert torch.allclose(new, th)


def test_max_threshold_clamp():
    """Strong push above max_threshold is clamped."""
    rule = _make(max_threshold=10.0, learning_rate=10.0)
    weights = torch.full((1, 4), 0.1)  # delta = 10*(0.5-0.1) = +4
    th = torch.tensor([8.0])
    new = rule.update(th, spike_times=torch.zeros(1), weights=weights)
    assert torch.allclose(new, torch.tensor([10.0]))


def test_learning_rate_decay():
    rule = _make(learning_rate=1.0, decay_factor=0.5)
    rule.learning_rate_step()
    assert rule.learning_rate == 0.5
