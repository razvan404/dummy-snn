import torch
import pytest

from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer
from spiking.learning.conv_learner import ConvLearner
from spiking.learning.multiplicative_stdp import MultiplicativeSTDP
from spiking.learning.stdp import STDP
from spiking.learning.wta import WinnerTakesAll
from spiking.threshold import (
    ConstantInitialization,
    CompetitiveThresholdAdaptation,
)
from spiking import iterate_spikes


def make_layer(in_channels=2, num_filters=4, kernel_size=3, padding=0, threshold=3.0):
    init = ConstantInitialization(threshold)
    return ConvIntegrateAndFireLayer(
        in_channels=in_channels,
        num_filters=num_filters,
        kernel_size=kernel_size,
        padding=padding,
        threshold_initialization=init,
        refractory_period=float("inf"),
    )


def make_multiplicative_stdp():
    return MultiplicativeSTDP(
        learning_rate=0.1,
        beta=1.0,
        w_min=0.0,
        w_max=1.0,
    )


def make_biological_stdp():
    return STDP(
        tau_pre=0.1,
        tau_post=0.1,
        max_pre_spike_time=1.0,
        learning_rate=0.1,
    )


def run_forward(layer, input_times):
    """Run forward pass to produce spikes in the layer."""
    layer.train()
    layer.reset()
    for incoming_spikes, current_time, dt in iterate_spikes(input_times):
        layer.forward(incoming_spikes, current_time, dt)


class TestConvLearnerConstruction:
    def test_create_with_all_components(self):
        layer = make_layer()
        stdp = make_multiplicative_stdp()
        wta = WinnerTakesAll()
        learner = ConvLearner(layer, stdp, competition=wta)
        assert learner.layer is layer
        assert learner.learning_mechanism is stdp
        assert learner.competition is wta

    def test_create_without_optional(self):
        layer = make_layer()
        stdp = make_multiplicative_stdp()
        learner = ConvLearner(layer, stdp)
        assert learner.competition is None
        assert learner.threshold_adaptation is None


class TestConvLearnerStep:
    def test_step_returns_float(self):
        torch.manual_seed(42)
        layer = make_layer(threshold=0.5)
        layer.weights.data.fill_(0.5)
        stdp = make_multiplicative_stdp()
        wta = WinnerTakesAll()
        learner = ConvLearner(layer, stdp, competition=wta)

        input_times = torch.rand(2, 8, 8)
        input_times[input_times > 0.7] = float("inf")
        run_forward(layer, input_times)

        dw = learner.step(input_times)
        assert isinstance(dw, float)

    def test_step_updates_weights(self):
        torch.manual_seed(42)
        layer = make_layer(threshold=0.5)
        layer.weights.data.fill_(0.5)
        stdp = make_multiplicative_stdp()
        wta = WinnerTakesAll()
        learner = ConvLearner(layer, stdp, competition=wta)

        input_times = torch.rand(2, 8, 8)
        input_times[input_times > 0.7] = float("inf")
        run_forward(layer, input_times)

        weights_before = layer.weights.data.clone()
        learner.step(input_times)

        # Only winning filter should change
        assert not torch.allclose(layer.weights.data, weights_before)

    def test_step_no_spikes_no_update(self):
        layer = make_layer(threshold=1000.0)  # high threshold, no spikes
        stdp = make_multiplicative_stdp()
        wta = WinnerTakesAll()
        learner = ConvLearner(layer, stdp, competition=wta)

        input_times = torch.rand(2, 8, 8)
        run_forward(layer, input_times)

        weights_before = layer.weights.data.clone()
        dw = learner.step(input_times)
        assert dw == 0.0
        assert torch.allclose(layer.weights.data, weights_before)

    def test_step_with_biological_stdp(self):
        torch.manual_seed(42)
        layer = make_layer(threshold=0.5)
        layer.weights.data.fill_(0.5)
        stdp = make_biological_stdp()
        wta = WinnerTakesAll()
        learner = ConvLearner(layer, stdp, competition=wta)

        input_times = torch.rand(2, 8, 8)
        input_times[input_times > 0.7] = float("inf")
        run_forward(layer, input_times)

        weights_before = layer.weights.data.clone()
        learner.step(input_times)
        assert not torch.allclose(layer.weights.data, weights_before)


class TestConvLearnerCompetition:
    def test_wta_selects_single_filter(self):
        torch.manual_seed(42)
        layer = make_layer(threshold=0.5)
        layer.weights.data.fill_(0.5)
        stdp = make_multiplicative_stdp()
        wta = WinnerTakesAll()
        learner = ConvLearner(layer, stdp, competition=wta)

        input_times = torch.rand(2, 8, 8)
        input_times[input_times > 0.7] = float("inf")
        run_forward(layer, input_times)
        learner.step(input_times)

        assert len(learner.neurons_to_learn) == 1

    def test_without_competition_all_spiked_learn(self):
        torch.manual_seed(42)
        layer = make_layer(threshold=0.5)
        layer.weights.data.fill_(0.5)
        stdp = make_multiplicative_stdp()
        learner = ConvLearner(layer, stdp)

        input_times = torch.rand(2, 8, 8)
        input_times[input_times > 0.7] = float("inf")
        run_forward(layer, input_times)
        learner.step(input_times)

        # All filters that spiked should learn
        assert len(learner.neurons_to_learn) >= 1


class TestConvLearnerThresholdAdaptation:
    def test_threshold_updated(self):
        torch.manual_seed(42)
        layer = make_layer(threshold=0.5)
        layer.weights.data.fill_(0.5)
        stdp = make_multiplicative_stdp()
        wta = WinnerTakesAll()
        adaptation = CompetitiveThresholdAdaptation(
            min_threshold=0.1,
            learning_rate=1.0,
            decay_factor=1.0,
        )
        learner = ConvLearner(
            layer, stdp, competition=wta, threshold_adaptation=adaptation
        )

        input_times = torch.rand(2, 8, 8)
        input_times[input_times > 0.7] = float("inf")
        run_forward(layer, input_times)

        thresholds_before = layer.thresholds.data.clone()
        learner.step(input_times)
        assert not torch.allclose(layer.thresholds.data, thresholds_before)


class TestConvLearnerLearningRateStep:
    def test_decays_both(self):
        layer = make_layer()
        stdp = make_multiplicative_stdp()
        adaptation = CompetitiveThresholdAdaptation(
            min_threshold=0.1,
            learning_rate=1.0,
            decay_factor=0.5,
        )
        learner = ConvLearner(layer, stdp, threshold_adaptation=adaptation)
        learner.learning_rate_step()
        assert adaptation.learning_rate == pytest.approx(0.5)
