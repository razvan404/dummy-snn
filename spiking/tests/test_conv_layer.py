import torch
import pytest

from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer
from spiking.threshold import NormalInitialization, ConstantInitialization
from spiking import iterate_spikes


def make_layer(
    in_channels=6,
    num_filters=4,
    kernel_size=5,
    stride=1,
    padding=0,
    threshold=5.0,
    refractory_period=1.0,
):
    threshold_init = ConstantInitialization(threshold)
    return ConvIntegrateAndFireLayer(
        in_channels=in_channels,
        num_filters=num_filters,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        threshold_initialization=threshold_init,
        refractory_period=refractory_period,
    )


class TestConvLayerConstruction:
    def test_weight_shape(self):
        layer = make_layer(in_channels=6, num_filters=4, kernel_size=5)
        assert layer.weights.shape == (4, 6, 5, 5)

    def test_threshold_shape(self):
        layer = make_layer(num_filters=8)
        assert layer.thresholds.shape == (8,)

    def test_weights_initialized_uniform(self):
        torch.manual_seed(42)
        layer = make_layer()
        assert (layer.weights >= 0).all()
        assert (layer.weights <= 1).all()

    def test_normal_threshold_initialization(self):
        init = NormalInitialization(avg_threshold=10.0, min_threshold=1.0, std_dev=0.1)
        layer = ConvIntegrateAndFireLayer(
            in_channels=3,
            num_filters=16,
            kernel_size=3,
            threshold_initialization=init,
        )
        assert layer.thresholds.shape == (16,)
        assert (layer.thresholds >= 1.0).all()


class TestConvLayerForward:
    def test_output_shape(self):
        layer = make_layer(in_channels=6, num_filters=4, kernel_size=5, padding=0)
        # Input: (6, 10, 10) → output spatial: (10-5+1, 10-5+1) = (6, 6)
        incoming = torch.ones(6, 10, 10)
        output = layer.forward(incoming, current_time=0.1, dt=0.1)
        assert output.shape == (4, 6, 6)

    def test_output_shape_with_padding(self):
        layer = make_layer(in_channels=6, num_filters=4, kernel_size=5, padding=2)
        incoming = torch.ones(6, 10, 10)
        output = layer.forward(incoming, current_time=0.1, dt=0.1)
        assert output.shape == (4, 10, 10)

    def test_output_shape_with_stride(self):
        layer = make_layer(in_channels=6, num_filters=4, kernel_size=5, stride=2)
        incoming = torch.ones(6, 10, 10)
        output = layer.forward(incoming, current_time=0.1, dt=0.1)
        assert output.shape == (4, 3, 3)

    def test_no_spikes_from_zero_input(self):
        layer = make_layer(threshold=5.0)
        incoming = torch.zeros(6, 10, 10)
        output = layer.forward(incoming, current_time=0.1, dt=0.1)
        assert (output == 0).all()

    def test_spike_times_recorded(self):
        """When input is strong enough, spike times should be recorded."""
        layer = make_layer(
            in_channels=1, num_filters=1, kernel_size=3, padding=1, threshold=0.1
        )
        # Set weights high to guarantee spiking
        layer.weights.data.fill_(1.0)
        incoming = torch.ones(1, 5, 5)
        layer.forward(incoming, current_time=0.3, dt=0.1)
        # At least some neurons should have spiked
        assert torch.isfinite(layer.spike_times).any()
        # Spiked neurons should have time 0.3
        spiked = layer.spike_times[torch.isfinite(layer.spike_times)]
        assert (spiked == 0.3).all()

    def test_spike_times_shape(self):
        layer = make_layer(in_channels=6, num_filters=4, kernel_size=5, padding=0)
        incoming = torch.ones(6, 10, 10)
        layer.forward(incoming, current_time=0.1, dt=0.1)
        assert layer.spike_times.shape == (4, 6, 6)


class TestConvLayerReset:
    def test_reset_clears_spike_times(self):
        layer = make_layer(
            in_channels=1, num_filters=1, kernel_size=3, padding=1, threshold=0.1
        )
        layer.weights.data.fill_(1.0)
        incoming = torch.ones(1, 5, 5)
        layer.forward(incoming, current_time=0.1, dt=0.1)
        assert torch.isfinite(layer.spike_times).any()
        layer.reset()
        assert torch.isinf(layer.spike_times).all()

    def test_reset_clears_membrane_potentials(self):
        layer = make_layer(threshold=1000.0)  # high threshold so no spikes
        incoming = torch.ones(6, 10, 10)
        layer.forward(incoming, current_time=0.1, dt=0.1)
        assert (layer.membrane_potentials != 0).any()
        layer.reset()
        assert (layer.membrane_potentials == 0).all()


class TestConvLayerRefractory:
    def test_refractory_prevents_double_firing(self):
        layer = make_layer(
            in_channels=1,
            num_filters=1,
            kernel_size=3,
            padding=1,
            threshold=0.1,
            refractory_period=float("inf"),
        )
        layer.weights.data.fill_(1.0)
        incoming = torch.ones(1, 5, 5)
        layer.forward(incoming, current_time=0.1, dt=0.1)
        first_spikes = layer.spike_times.clone()

        # Second forward — spiked neurons should stay refractory
        layer.forward(incoming, current_time=0.2, dt=0.1)
        assert (layer.spike_times == first_spikes).all()


class TestConvLayerAnalyticalInference:
    def test_infer_spike_times_shape(self):
        layer = make_layer(in_channels=6, num_filters=4, kernel_size=5, padding=0)
        input_times = torch.rand(6, 10, 10)
        result = layer.infer_spike_times(input_times)
        assert result.shape == (4, 6, 6)

    def test_infer_spike_times_batch_shape(self):
        layer = make_layer(in_channels=6, num_filters=4, kernel_size=5, padding=0)
        input_times = torch.rand(8, 6, 10, 10)
        result = layer.infer_spike_times_batch(input_times)
        assert result.shape == (8, 4, 6, 6)

    def test_infer_matches_forward_pass(self):
        """Analytical inference should match step-by-step forward pass."""
        torch.manual_seed(42)
        layer = make_layer(
            in_channels=2,
            num_filters=2,
            kernel_size=3,
            padding=0,
            threshold=3.0,
            refractory_period=float("inf"),
        )
        # Create input spike times with some inf (non-spiking)
        input_times = torch.rand(2, 8, 8)
        input_times[input_times > 0.7] = float("inf")

        # Analytical inference
        analytical = layer.infer_spike_times(input_times)

        # Step-by-step forward pass
        layer.eval()
        layer.reset()
        for incoming_spikes, current_time, dt in iterate_spikes(input_times):
            layer.forward(incoming_spikes, current_time, dt)
        forward_times = layer.spike_times.clone()

        # They should match
        both_finite = torch.isfinite(analytical) & torch.isfinite(forward_times)
        if both_finite.any():
            torch.testing.assert_close(
                analytical[both_finite],
                forward_times[both_finite],
                atol=1e-5,
                rtol=1e-5,
            )
        # Same inf pattern
        assert (torch.isinf(analytical) == torch.isinf(forward_times)).all()

    def test_all_inf_input(self):
        layer = make_layer(in_channels=6, num_filters=4, kernel_size=5)
        input_times = torch.full((6, 10, 10), float("inf"))
        result = layer.infer_spike_times(input_times)
        assert torch.isinf(result).all()

    def test_all_inf_input_batch(self):
        layer = make_layer(in_channels=6, num_filters=4, kernel_size=5)
        input_times = torch.full((3, 6, 10, 10), float("inf"))
        result = layer.infer_spike_times_batch(input_times)
        assert torch.isinf(result).all()
