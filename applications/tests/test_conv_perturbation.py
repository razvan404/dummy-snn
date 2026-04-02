import numpy as np
import torch
import pytest

from spiking import ConvIntegrateAndFireLayer, ConstantInitialization
from spiking.evaluation.conv_feature_extraction import sum_pool_features
from spiking.evaluation.feature_extraction import spike_times_to_features


def _make_tiny_conv_layer(
    in_channels=2, num_filters=4, kernel_size=3, image_size=8
):
    """Create a small ConvIntegrateAndFireLayer for testing."""
    torch.manual_seed(42)
    layer = ConvIntegrateAndFireLayer(
        in_channels=in_channels,
        num_filters=num_filters,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
        threshold_initialization=ConstantInitialization(threshold=3.0),
        refractory_period=float("inf"),
    )
    torch.nn.init.uniform_(layer.weights, 0.0, 1.0)
    return layer


def _make_synthetic_input(batch_size=10, in_channels=2, image_size=8):
    """Create synthetic spike times in [0, 1] with some inf (no-spike) values."""
    torch.manual_seed(123)
    times = torch.rand(batch_size, in_channels, image_size, image_size)
    # Set ~20% of inputs to inf (no spike)
    mask = torch.rand_like(times) < 0.2
    times[mask] = float("inf")
    return times


class TestMultiThresholdConvAccumulate:
    """Test that multi_threshold_conv_accumulate matches naive per-threshold inference."""

    def test_matches_naive_single_threshold(self):
        """Single threshold should match _conv2d_accumulate exactly."""
        from applications.threshold_research.conv_neuron_perturbation import (
            multi_threshold_conv_accumulate,
        )

        layer = _make_tiny_conv_layer()
        input_times = _make_synthetic_input(batch_size=5)

        # Naive: use layer's own inference
        naive_result = layer.infer_spike_times_batch(input_times)  # (B, F, oH, oW)

        # Multi-threshold with original thresholds (1 fraction)
        thresholds_2d = layer.thresholds.detach().unsqueeze(0)  # (1, F)
        result = multi_threshold_conv_accumulate(
            input_times,
            layer.weights_4d.detach(),
            thresholds_2d,
            stride=layer.stride,
            padding=layer.padding,
        )
        assert result.shape == (1, 5, 4, 6, 6)
        torch.testing.assert_close(result[0], naive_result)

    def test_matches_naive_multiple_thresholds(self):
        """All 31 perturbation fractions should match running _conv2d_accumulate 31 times."""
        from applications.threshold_research.conv_neuron_perturbation import (
            multi_threshold_conv_accumulate,
        )

        layer = _make_tiny_conv_layer()
        input_times = _make_synthetic_input(batch_size=8)
        original_thresholds = layer.thresholds.detach().clone()

        perturbation_fractions = [round(-0.5 + i * 0.025, 3) for i in range(31)]

        # Build threshold matrix
        thresholds_2d = torch.stack(
            [original_thresholds * (1.0 + frac) for frac in perturbation_fractions]
        )

        # Multi-threshold single pass
        result = multi_threshold_conv_accumulate(
            input_times,
            layer.weights_4d.detach(),
            thresholds_2d,
            stride=layer.stride,
            padding=layer.padding,
        )
        assert result.shape[0] == 31

        # Naive: run inference 31 times with different thresholds
        for frac_idx, frac in enumerate(perturbation_fractions):
            layer.thresholds.data = original_thresholds * (1.0 + frac)
            naive = layer.infer_spike_times_batch(input_times)
            torch.testing.assert_close(
                result[frac_idx], naive,
                msg=f"Mismatch at frac={frac}",
            )

        # Restore original thresholds
        layer.thresholds.data = original_thresholds

    def test_all_inf_input(self):
        """All-inf input should produce all-inf output."""
        from applications.threshold_research.conv_neuron_perturbation import (
            multi_threshold_conv_accumulate,
        )

        layer = _make_tiny_conv_layer()
        input_times = torch.full((3, 2, 8, 8), float("inf"))
        thresholds_2d = layer.thresholds.detach().unsqueeze(0)

        result = multi_threshold_conv_accumulate(
            input_times, layer.weights_4d.detach(), thresholds_2d,
            stride=layer.stride, padding=layer.padding,
        )
        assert torch.isinf(result).all()

    def test_early_exit(self):
        """Very low thresholds should cause early exit (all spike at first time)."""
        from applications.threshold_research.conv_neuron_perturbation import (
            multi_threshold_conv_accumulate,
        )

        layer = _make_tiny_conv_layer()
        input_times = _make_synthetic_input(batch_size=3)

        # Very low thresholds — everything should spike immediately
        thresholds_2d = torch.full((1, layer.num_filters), 0.01)
        result = multi_threshold_conv_accumulate(
            input_times, layer.weights_4d.detach(), thresholds_2d,
            stride=layer.stride, padding=layer.padding,
        )
        # All outputs should have spiked (no inf)
        assert torch.isfinite(result).all()


class TestSpikeTimesToPooledFeatures:
    def test_output_shape(self):
        from applications.threshold_research.conv_neuron_perturbation import (
            _spike_times_to_pooled_features,
        )

        spike_times = torch.rand(3, 5, 4, 6, 6)  # (fracs, B, F, oH, oW)
        result = _spike_times_to_pooled_features(spike_times, t_target=None, pool_size=2)
        # pool_size=2 on 6x6 → 3x3 grid, but sum_pool divides into 2x2 regions
        # 6 // 2 = 3 per region, so pooled shape: (F, 2, 2) → flat 4*2*2 = 16
        assert result.shape == (3, 5, 4 * 2 * 2)

    def test_no_pool(self):
        from applications.threshold_research.conv_neuron_perturbation import (
            _spike_times_to_pooled_features,
        )

        spike_times = torch.rand(2, 4, 3, 6, 6)
        result = _spike_times_to_pooled_features(spike_times, t_target=None, pool_size=1)
        assert result.shape == (2, 4, 3 * 6 * 6)


class TestFeatureEquivalence:
    """Verify that frac=0.0 perturbed features match baseline."""

    def test_zero_fraction_matches_baseline(self):
        from applications.threshold_research.conv_neuron_perturbation import (
            multi_threshold_conv_accumulate,
            _spike_times_to_pooled_features,
        )

        layer = _make_tiny_conv_layer()
        input_times = _make_synthetic_input(batch_size=10)
        pool_size = 2

        # Baseline: layer's own inference
        baseline_st = layer.infer_spike_times_batch(input_times)
        baseline_feat = spike_times_to_features(baseline_st, t_target=None)
        baseline_pooled = sum_pool_features(baseline_feat, pool_size).flatten(1).numpy()

        # Multi-threshold with frac=0.0 only
        thresholds_2d = layer.thresholds.detach().unsqueeze(0)  # (1, F)
        result_st = multi_threshold_conv_accumulate(
            input_times, layer.weights_4d.detach(), thresholds_2d,
            stride=layer.stride, padding=layer.padding,
        )
        result_feat = _spike_times_to_pooled_features(result_st, t_target=None, pool_size=pool_size)

        np.testing.assert_allclose(result_feat[0], baseline_pooled, rtol=1e-5)


class TestEvaluateConvPerturbations:
    """Sanity checks for the evaluation phase."""

    def test_identity_fraction_matches_baseline_accuracy(self):
        """When perturbed == baseline, accuracy should equal baseline."""
        from applications.threshold_research.conv_neuron_perturbation import (
            evaluate_conv_perturbations,
        )

        num_filters = 4
        pool_size = 2
        cols = num_filters * pool_size * pool_size
        N_train, N_val = 50, 20

        np.random.seed(42)
        X_train = np.random.rand(N_train, cols).astype(np.float32)
        X_val = np.random.rand(N_val, cols).astype(np.float32)
        y_train = np.random.randint(0, 3, N_train)
        y_val = np.random.randint(0, 3, N_val)

        # perturbed == baseline for all fractions
        perturbed_train = np.tile(X_train, (1, 1, 1)).reshape(1, N_train, cols)
        perturbed_val = np.tile(X_val, (1, 1, 1)).reshape(1, N_val, cols)

        features = {
            "baseline_train": X_train,
            "baseline_val": X_val,
            "labels_train": y_train,
            "labels_val": y_val,
            "perturbed_train": perturbed_train,
            "perturbed_val": perturbed_val,
            "original_thresholds": [1.0] * num_filters,
            "perturbation_fractions": [0.0],
        }

        result = evaluate_conv_perturbations(
            features=features,
            num_filters=num_filters,
            pool_size=pool_size,
        )

        baseline_acc = result["baseline"]["accuracy"]
        # All filter perturbations at frac=0.0 should give baseline accuracy
        for f in range(num_filters):
            assert result["accuracy_matrix"][f][0] == pytest.approx(
                baseline_acc
            ), f"Filter {f} accuracy mismatch"
