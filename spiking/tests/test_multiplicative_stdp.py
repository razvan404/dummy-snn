import torch
import pytest

from spiking.learning.multiplicative_stdp import MultiplicativeSTDP


def make_stdp(**kwargs):
    defaults = dict(
        learning_rate=0.1,
        decay_factor=1.0,
        beta=1.0,
        w_min=0.0,
        w_max=1.0,
        t_ltp=float("inf"),
    )
    defaults.update(kwargs)
    return MultiplicativeSTDP(**defaults)


class TestMultiplicativeSTDPPotentiation:
    def test_causal_pair_increases_weight(self):
        """Pre before post (delta_t > 0) → potentiation."""
        stdp = make_stdp()
        weights = torch.tensor([[0.5]])
        pre = torch.tensor([0.2])  # pre spike time
        post = torch.tensor([[0.5]])  # post spike time (after pre)
        updated = stdp.update_weights(weights, pre, post)
        assert updated[0, 0] > weights[0, 0]

    def test_potentiation_larger_near_wmin(self):
        """Weights near w_min should get larger potentiation."""
        stdp = make_stdp(beta=2.0)
        pre = torch.tensor([0.2])
        post = torch.tensor([[0.5]])

        w_low = torch.tensor([[0.1]])
        w_high = torch.tensor([[0.8]])
        dw_low = stdp.update_weights(w_low, pre, post)[0, 0] - w_low[0, 0]
        dw_high = stdp.update_weights(w_high, pre, post)[0, 0] - w_high[0, 0]
        assert dw_low > dw_high


class TestMultiplicativeSTDPDepression:
    def test_anticausal_pair_decreases_weight(self):
        """Post before pre (delta_t < 0) → depression."""
        stdp = make_stdp()
        weights = torch.tensor([[0.5]])
        pre = torch.tensor([0.8])  # pre spike time (after post)
        post = torch.tensor([[0.3]])  # post spike time
        updated = stdp.update_weights(weights, pre, post)
        assert updated[0, 0] < weights[0, 0]

    def test_depression_larger_near_wmax(self):
        """Weights near w_max should get larger depression."""
        stdp = make_stdp(beta=2.0)
        pre = torch.tensor([0.8])
        post = torch.tensor([[0.3]])

        w_low = torch.tensor([[0.2]])
        w_high = torch.tensor([[0.9]])
        dw_low = abs(
            stdp.update_weights(w_low, pre, post)[0, 0].item() - w_low[0, 0].item()
        )
        dw_high = abs(
            stdp.update_weights(w_high, pre, post)[0, 0].item() - w_high[0, 0].item()
        )
        assert dw_high > dw_low


class TestMultiplicativeSTDPBounds:
    def test_weight_stays_above_wmin(self):
        stdp = make_stdp(learning_rate=10.0)
        weights = torch.tensor([[0.01]])
        pre = torch.tensor([0.8])
        post = torch.tensor([[0.1]])  # depression
        updated = stdp.update_weights(weights, pre, post)
        assert updated[0, 0] >= 0.0

    def test_weight_stays_below_wmax(self):
        stdp = make_stdp(learning_rate=10.0)
        weights = torch.tensor([[0.99]])
        pre = torch.tensor([0.1])
        post = torch.tensor([[0.5]])  # potentiation
        updated = stdp.update_weights(weights, pre, post)
        assert updated[0, 0] <= 1.0


class TestMultiplicativeSTDPTltp:
    def test_pair_beyond_tltp_causes_depression(self):
        """If pre-post delay exceeds t_ltp, treat as depression."""
        stdp = make_stdp(t_ltp=0.1)
        weights = torch.tensor([[0.5]])
        pre = torch.tensor([0.1])
        post = torch.tensor([[0.5]])  # delta = 0.4 > t_ltp = 0.1
        updated = stdp.update_weights(weights, pre, post)
        assert updated[0, 0] < weights[0, 0]

    def test_pair_within_tltp_potentiates(self):
        stdp = make_stdp(t_ltp=0.5)
        weights = torch.tensor([[0.5]])
        pre = torch.tensor([0.3])
        post = torch.tensor([[0.5]])  # delta = 0.2 < t_ltp
        updated = stdp.update_weights(weights, pre, post)
        assert updated[0, 0] > weights[0, 0]


class TestMultiplicativeSTDPMultipleWeights:
    def test_batch_update(self):
        """Multiple neurons × multiple inputs."""
        stdp = make_stdp()
        weights = torch.rand(3, 5)
        pre = torch.rand(5)
        post = torch.rand(3, 1)
        updated = stdp.update_weights(weights, pre, post)
        assert updated.shape == (3, 5)
        # Weights should have changed
        assert not torch.allclose(updated, weights)

    def test_inf_pre_spike_causes_depression(self):
        """Non-spiking pre-synaptic → depression."""
        stdp = make_stdp()
        weights = torch.tensor([[0.5]])
        pre = torch.tensor([float("inf")])
        post = torch.tensor([[0.3]])
        updated = stdp.update_weights(weights, pre, post)
        assert updated[0, 0] < weights[0, 0]


class TestMultiplicativeSTDPLearningRateDecay:
    def test_decay_reduces_learning_rate(self):
        stdp = make_stdp(learning_rate=0.1, decay_factor=0.5)
        stdp.learning_rate_step()
        assert stdp.learning_rate == pytest.approx(0.05)

    def test_no_decay_when_factor_is_one(self):
        stdp = make_stdp(learning_rate=0.1, decay_factor=1.0)
        stdp.learning_rate_step()
        assert stdp.learning_rate == pytest.approx(0.1)
