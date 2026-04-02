import torch
import pytest

from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer
from spiking.layers.integrate_and_fire import IntegrateAndFireLayer
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
    def test_weight_shape_2d(self):
        layer = make_layer(in_channels=6, num_filters=4, kernel_size=5)
        assert layer.weights.shape == (4, 6 * 5 * 5)

    def test_weights_4d_property(self):
        layer = make_layer(in_channels=6, num_filters=4, kernel_size=5)
        assert layer.weights_4d.shape == (4, 6, 5, 5)
        # Should be a view (same data)
        assert layer.weights_4d.data_ptr() == layer.weights.data_ptr()

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

    def test_inherits_from_integrate_and_fire(self):
        layer = make_layer()
        assert isinstance(layer, IntegrateAndFireLayer)


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


class TestConvFCEquivalence:
    """Verify ConvIntegrateAndFireLayer matches IntegrateAndFireLayer
    when applied to the same input with the same weights.

    Since ConvIntegrateAndFireLayer now inherits from IntegrateAndFireLayer
    and delegates to super().infer_spike_times_batch(), equivalence is
    structural — but we still verify it end-to-end.
    """

    def test_5x5_kernel_on_5x5_image(self):
        """Single 5x5 patch: conv output is 1x1, must match FC layer exactly."""
        C, K, F = 6, 5, 8
        torch.manual_seed(0)

        # Create conv layer
        conv_init = ConstantInitialization(10.0)
        conv_layer = ConvIntegrateAndFireLayer(
            in_channels=C, num_filters=F, kernel_size=K,
            threshold_initialization=conv_init, refractory_period=float("inf"),
        )

        # Create FC layer sharing the same weights (already 2D)
        fc_init = ConstantInitialization(10.0)
        fc_layer = IntegrateAndFireLayer(
            num_inputs=C * K * K, num_outputs=F,
            threshold_initialization=fc_init, refractory_period=float("inf"),
        )
        fc_layer.weights.data.copy_(conv_layer.weights.data)
        fc_layer.thresholds.data.copy_(conv_layer.thresholds.data)

        # Input: (C, 5, 5) → conv output is (F, 1, 1)
        input_times = torch.rand(C, K, K)
        input_times = (input_times * 16).floor() / 16
        input_times[input_times >= 1.0] = float("inf")

        conv_result = conv_layer.infer_spike_times(input_times)  # (F, 1, 1)
        fc_result = fc_layer.infer_spike_times(input_times.flatten())  # (F,)

        assert conv_result.shape == (F, 1, 1)
        assert fc_result.shape == (F,)
        assert torch.allclose(conv_result.squeeze(), fc_result), (
            f"Conv and FC outputs differ:\n"
            f"  conv: {conv_result.squeeze()}\n"
            f"  fc:   {fc_result}"
        )

    def test_5x5_kernel_on_7x7_image_3x3_positions(self):
        """7x7 image with 5x5 kernel gives 3x3 output. Each spatial position
        must match the FC layer applied to the corresponding flattened patch."""
        C, K, F = 6, 5, 8
        H, W = 7, 7
        oH, oW = H - K + 1, W - K + 1  # 3, 3
        torch.manual_seed(1)

        conv_init = ConstantInitialization(10.0)
        conv_layer = ConvIntegrateAndFireLayer(
            in_channels=C, num_filters=F, kernel_size=K,
            threshold_initialization=conv_init, refractory_period=float("inf"),
        )

        fc_init = ConstantInitialization(10.0)
        fc_layer = IntegrateAndFireLayer(
            num_inputs=C * K * K, num_outputs=F,
            threshold_initialization=fc_init, refractory_period=float("inf"),
        )
        fc_layer.weights.data.copy_(conv_layer.weights.data)
        fc_layer.thresholds.data.copy_(conv_layer.thresholds.data)

        # Input: (C, 7, 7)
        input_times = torch.rand(C, H, W)
        input_times = (input_times * 16).floor() / 16
        input_times[input_times >= 1.0] = float("inf")

        conv_result = conv_layer.infer_spike_times(input_times)  # (F, 3, 3)
        assert conv_result.shape == (F, oH, oW)

        # Check each of the 3x3 spatial positions against FC layer
        for r in range(oH):
            for c in range(oW):
                patch = input_times[:, r : r + K, c : c + K].flatten()
                fc_result = fc_layer.infer_spike_times(patch)  # (F,)
                conv_at_pos = conv_result[:, r, c]
                assert torch.allclose(conv_at_pos, fc_result), (
                    f"Mismatch at position ({r},{c}):\n"
                    f"  conv: {conv_at_pos}\n"
                    f"  fc:   {fc_result}"
                )

    def test_batch_5x5_kernel_on_7x7_image(self):
        """Batch version: same test as above but using infer_spike_times_batch."""
        C, K, F = 6, 5, 8
        H, W = 7, 7
        B = 4
        torch.manual_seed(2)

        conv_init = ConstantInitialization(10.0)
        conv_layer = ConvIntegrateAndFireLayer(
            in_channels=C, num_filters=F, kernel_size=K,
            threshold_initialization=conv_init, refractory_period=float("inf"),
        )

        fc_init = ConstantInitialization(10.0)
        fc_layer = IntegrateAndFireLayer(
            num_inputs=C * K * K, num_outputs=F,
            threshold_initialization=fc_init, refractory_period=float("inf"),
        )
        fc_layer.weights.data.copy_(conv_layer.weights.data)
        fc_layer.thresholds.data.copy_(conv_layer.thresholds.data)

        input_times = torch.rand(B, C, H, W)
        input_times = (input_times * 16).floor() / 16
        input_times[input_times >= 1.0] = float("inf")

        conv_result = conv_layer.infer_spike_times_batch(input_times)  # (B, F, 3, 3)

        # Verify each batch element and spatial position
        oH, oW = H - K + 1, W - K + 1
        for b in range(B):
            for r in range(oH):
                for c in range(oW):
                    patch = input_times[b, :, r : r + K, c : c + K].flatten()
                    fc_result = fc_layer.infer_spike_times(patch)
                    assert torch.allclose(conv_result[b, :, r, c], fc_result), (
                        f"Mismatch at batch={b}, pos=({r},{c})"
                    )


class TestConv2dVsUnfoldEquivalence:
    """Verify conv2d-based and unfold-based batch inference produce identical results."""

    def _make_input(self, B, C, H, W, seed=42):
        torch.manual_seed(seed)
        t = torch.rand(B, C, H, W)
        t = (t * 16).floor() / 16
        t[t >= 0.9] = float("inf")
        return t

    def test_small_no_padding(self):
        layer = make_layer(in_channels=2, num_filters=4, kernel_size=3, padding=0)
        inp = self._make_input(4, 2, 8, 8)
        conv2d_result = layer.infer_spike_times_batch(inp)
        unfold_result = layer.infer_spike_times_batch_unfold(inp)
        torch.testing.assert_close(conv2d_result, unfold_result)

    def test_with_padding(self):
        layer = make_layer(
            in_channels=6, num_filters=8, kernel_size=5, padding=2, threshold=5.0
        )
        inp = self._make_input(8, 6, 10, 10, seed=7)
        conv2d_result = layer.infer_spike_times_batch(inp)
        unfold_result = layer.infer_spike_times_batch_unfold(inp)
        torch.testing.assert_close(conv2d_result, unfold_result)

    def test_large_batch(self):
        layer = make_layer(in_channels=6, num_filters=16, kernel_size=5, padding=0)
        inp = self._make_input(32, 6, 32, 32, seed=99)
        conv2d_result = layer.infer_spike_times_batch(inp)
        unfold_result = layer.infer_spike_times_batch_unfold(inp)
        torch.testing.assert_close(conv2d_result, unfold_result)

    def test_all_inf(self):
        layer = make_layer(in_channels=6, num_filters=4, kernel_size=5)
        inp = torch.full((4, 6, 10, 10), float("inf"))
        conv2d_result = layer.infer_spike_times_batch(inp)
        unfold_result = layer.infer_spike_times_batch_unfold(inp)
        assert torch.isinf(conv2d_result).all()
        assert torch.isinf(unfold_result).all()

    def test_single_image_matches(self):
        """infer_spike_times (single, unfold) must match infer_spike_times_batch (conv2d)."""
        layer = make_layer(in_channels=6, num_filters=8, kernel_size=5, padding=0)
        inp = self._make_input(1, 6, 12, 12, seed=3)
        batch_result = layer.infer_spike_times_batch(inp)  # (1, F, oH, oW)
        single_result = layer.infer_spike_times(inp.squeeze(0))  # (F, oH, oW)
        torch.testing.assert_close(batch_result.squeeze(0), single_result)


class TestConvInferenceBenchmark:
    """Benchmark conv2d vs unfold approaches on 64x64 images with 5x5 kernel."""

    def test_benchmark_64x64(self):
        import time

        C, F_n, K = 6, 64, 5
        H, W, B = 64, 64, 16
        torch.manual_seed(42)

        layer = make_layer(
            in_channels=C, num_filters=F_n, kernel_size=K, threshold=10.0
        )
        inp = torch.rand(B, C, H, W)
        inp = (inp * 16).floor() / 16
        inp[inp >= 0.9] = float("inf")

        warmup = 3
        runs = 10

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                layer.infer_spike_times_batch(inp)
                layer.infer_spike_times_batch_unfold(inp)

        # Benchmark conv2d
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(runs):
                layer.infer_spike_times_batch(inp)
            t_conv2d = (time.perf_counter() - t0) / runs

        # Benchmark unfold
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(runs):
                layer.infer_spike_times_batch_unfold(inp)
            t_unfold = (time.perf_counter() - t0) / runs

        # Verify they match
        with torch.no_grad():
            r1 = layer.infer_spike_times_batch(inp)
            r2 = layer.infer_spike_times_batch_unfold(inp)
        torch.testing.assert_close(r1, r2)

        oH, oW = layer._compute_output_size(H, W)
        print(f"\n{'='*60}")
        print(f"Benchmark: {B}x{C}x{H}x{W} input, {F_n} filters, {K}x{K} kernel")
        print(f"Output spatial: {oH}x{oW} = {oH*oW} positions")
        print(f"  conv2d:  {t_conv2d*1000:.2f} ms")
        print(f"  unfold:  {t_unfold*1000:.2f} ms")
        print(f"  speedup: {t_unfold/t_conv2d:.2f}x")
        print(f"{'='*60}")
