import json
import os

import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader

from applications.datasets import MnistSubsetDataset


IMAGE_SHAPE = (8, 8)
SPIKE_SHAPE = (2, *IMAGE_SHAPE)
NUM_INPUTS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * 2  # 128


@pytest.fixture(scope="module")
def train_loader():
    dataset = MnistSubsetDataset("data/mnist-subset", "train", image_shape=IMAGE_SHAPE)
    return DataLoader(dataset, batch_size=None, shuffle=True)


@pytest.fixture(scope="module")
def val_loader():
    dataset = MnistSubsetDataset("data/mnist-subset", "test", image_shape=IMAGE_SHAPE)
    return DataLoader(dataset, batch_size=None, shuffle=False)


def _make_tiny_layer(num_inputs=8, num_outputs=4):
    """Create a small IntegrateAndFireLayer for testing."""
    from spiking import IntegrateAndFireLayer, ConstantInitialization

    torch.manual_seed(42)
    layer = IntegrateAndFireLayer(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        threshold_initialization=ConstantInitialization(threshold=5.0),
        refractory_period=float("inf"),
    )
    return layer


class TestPrecomputeCumulativePotentials:
    def test_output_shapes(self):
        from applications.threshold_research.neuron_perturbation import (
            precompute_cumulative_potentials,
        )

        layer = _make_tiny_layer(num_inputs=8, num_outputs=4)
        # 3 samples, 8 inputs
        input_times = torch.tensor(
            [
                [
                    0.1,
                    0.2,
                    0.3,
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    float("inf"),
                ],
                [
                    0.1,
                    0.1,
                    0.5,
                    0.5,
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    float("inf"),
                ],
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ]
        )

        cum_potentials, boundary_times = precompute_cumulative_potentials(
            input_times, layer.weights
        )
        B = input_times.shape[0]
        O = layer.num_outputs
        G = len(boundary_times)

        assert cum_potentials.shape == (B, O, G)
        assert boundary_times.shape == (G,)
        # boundary_times should be sorted
        assert (boundary_times[1:] >= boundary_times[:-1]).all()

    def test_values_match_manual_computation(self):
        from applications.threshold_research.neuron_perturbation import (
            precompute_cumulative_potentials,
        )

        layer = _make_tiny_layer(num_inputs=4, num_outputs=2)
        # Single sample: inputs at times 0.1 and 0.3 (indices 0 and 2)
        input_times = torch.tensor([[0.1, float("inf"), 0.3, float("inf")]])
        weights = layer.weights.detach()  # (2, 4)

        cum_potentials, boundary_times = precompute_cumulative_potentials(
            input_times, weights
        )

        # At t=0.1, only input 0 fires → contribution = weights[:, 0]
        # At t=0.3, input 2 also fires → cumulative += weights[:, 2]
        assert boundary_times.tolist() == pytest.approx([0.1, 0.3])

        expected_t1 = weights[:, 0].numpy()
        expected_t2 = (weights[:, 0] + weights[:, 2]).numpy()

        np.testing.assert_allclose(
            cum_potentials[0, :, 0].numpy(), expected_t1, atol=1e-6
        )
        np.testing.assert_allclose(
            cum_potentials[0, :, 1].numpy(), expected_t2, atol=1e-6
        )

    def test_all_infinite_inputs(self):
        from applications.threshold_research.neuron_perturbation import (
            precompute_cumulative_potentials,
        )

        layer = _make_tiny_layer(num_inputs=4, num_outputs=2)
        input_times = torch.full((2, 4), float("inf"))

        cum_potentials, boundary_times = precompute_cumulative_potentials(
            input_times, layer.weights
        )
        assert boundary_times.shape[0] == 0
        assert cum_potentials.shape == (2, 2, 0)


class TestSpikeTimesFromPotentials:
    def test_finds_correct_crossing_time(self):
        from applications.threshold_research.neuron_perturbation import (
            spike_times_from_potentials,
        )

        # 2 samples, 3 boundary times
        cum_potentials = torch.tensor(
            [
                [1.0, 4.0, 7.0],  # crosses threshold=5.0 at index 2
                [2.0, 6.0, 8.0],  # crosses threshold=5.0 at index 1
            ]
        )
        boundary_times = torch.tensor([0.1, 0.3, 0.5])
        threshold = 5.0

        result = spike_times_from_potentials(cum_potentials, boundary_times, threshold)
        assert result.shape == (2,)
        assert result[0].item() == pytest.approx(0.5)
        assert result[1].item() == pytest.approx(0.3)

    def test_no_crossing_returns_inf(self):
        from applications.threshold_research.neuron_perturbation import (
            spike_times_from_potentials,
        )

        cum_potentials = torch.tensor(
            [
                [1.0, 2.0, 3.0],  # never reaches threshold=10.0
            ]
        )
        boundary_times = torch.tensor([0.1, 0.3, 0.5])
        threshold = 10.0

        result = spike_times_from_potentials(cum_potentials, boundary_times, threshold)
        assert torch.isinf(result[0])

    def test_empty_potentials(self):
        from applications.threshold_research.neuron_perturbation import (
            spike_times_from_potentials,
        )

        cum_potentials = torch.zeros((3, 0))
        boundary_times = torch.zeros(0)
        threshold = 5.0

        result = spike_times_from_potentials(cum_potentials, boundary_times, threshold)
        assert result.shape == (3,)
        assert torch.isinf(result).all()


class TestConsistencyWithInferSpikeTimesBatch:
    """Verify precomputed approach gives identical spike times to infer_spike_times_batch."""

    def test_spike_times_match(self):
        from applications.threshold_research.neuron_perturbation import (
            precompute_cumulative_potentials,
            spike_times_from_potentials,
        )

        torch.manual_seed(123)
        layer = _make_tiny_layer(num_inputs=16, num_outputs=8)
        B = 10
        # Random input times with some infinities
        input_times = torch.rand(B, 16)
        mask = torch.rand(B, 16) > 0.5
        input_times[mask] = float("inf")

        # Reference: use layer's built-in method
        expected = layer.infer_spike_times_batch(input_times)

        # Precomputed approach
        cum_potentials, boundary_times = precompute_cumulative_potentials(
            input_times, layer.weights
        )

        # Reconstruct spike times for each neuron
        result = torch.full_like(expected, float("inf"))
        for neuron_idx in range(layer.num_outputs):
            threshold = layer.thresholds[neuron_idx].item()
            neuron_potentials = cum_potentials[:, neuron_idx, :]  # (B, G)
            result[:, neuron_idx] = spike_times_from_potentials(
                neuron_potentials, boundary_times, threshold
            )

        # They must match exactly (same algorithm, just restructured)
        torch.testing.assert_close(result, expected)

    def test_spike_times_match_with_real_data(self, train_loader):
        """Consistency check on actual MNIST subset data."""
        from applications.threshold_research.neuron_perturbation import (
            precompute_cumulative_potentials,
            spike_times_from_potentials,
        )
        from applications.deep_linear.model import create_model

        setup = {
            "threshold_init": {
                "avg_threshold": 5.0,
                "min_threshold": 1.0,
                "std_dev": 0.5,
            },
        }
        torch.manual_seed(42)
        model = create_model(setup, NUM_INPUTS, [16])
        layer = model.layers[0]

        # Collect a small batch
        samples = []
        for i, (times, _label) in enumerate(train_loader):
            samples.append(times.flatten())
            if i >= 19:
                break
        input_times = torch.stack(samples)

        expected = layer.infer_spike_times_batch(input_times)

        cum_potentials, boundary_times = precompute_cumulative_potentials(
            input_times, layer.weights
        )

        result = torch.full_like(expected, float("inf"))
        for neuron_idx in range(layer.num_outputs):
            threshold = layer.thresholds[neuron_idx].item()
            neuron_potentials = cum_potentials[:, neuron_idx, :]
            result[:, neuron_idx] = spike_times_from_potentials(
                neuron_potentials, boundary_times, threshold
            )

        torch.testing.assert_close(result, expected)


class TestWeightScalingSweep:
    def test_output_structure(self, train_loader, val_loader):
        from applications.threshold_research.weight_scaling import weight_scaling_sweep
        from applications.deep_linear.model import create_model
        from spiking import save_model

        setup = {
            "threshold_init": {
                "avg_threshold": 5.0,
                "min_threshold": 1.0,
                "std_dev": 0.5,
            },
        }
        torch.manual_seed(42)
        model = create_model(setup, NUM_INPUTS, [16])

        model_path = "/tmp/test_weight_scaling_model.pth"
        save_model(model, model_path)

        result = weight_scaling_sweep(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            scale_factors=[0.9, 1.1],
        )

        assert "baseline" in result
        assert "factors" in result
        assert "accuracy" in result["baseline"]
        assert 0.9 in result["factors"]
        assert 1.1 in result["factors"]
        assert "accuracy" in result["factors"][0.9]

    def test_baseline_matches_independent_eval(self, train_loader, val_loader):
        from applications.threshold_research.weight_scaling import weight_scaling_sweep
        from applications.common import evaluate_model
        from applications.deep_linear.model import create_model
        from spiking import save_model
        from spiking.layers import SpikingSequential

        setup = {
            "threshold_init": {
                "avg_threshold": 5.0,
                "min_threshold": 1.0,
                "std_dev": 0.5,
            },
        }
        torch.manual_seed(42)
        model = create_model(setup, NUM_INPUTS, [16])

        model_path = "/tmp/test_ws_baseline_model.pth"
        save_model(model, model_path)

        result = weight_scaling_sweep(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            scale_factors=[0.9],
        )

        # Independent baseline evaluation
        torch.manual_seed(42)
        model2 = create_model(setup, NUM_INPUTS, [16])
        sub_model = SpikingSequential(*model2.layers[:1])
        _, val_m = evaluate_model(sub_model, train_loader, val_loader, SPIKE_SHAPE)

        assert result["baseline"]["accuracy"] == pytest.approx(
            val_m["accuracy"], abs=1e-6
        )


class TestDiffFactors:
    def test_no_existing_returns_all_desired(self):
        from applications.threshold_research.weight_scaling import _diff_factors

        missing, stale = _diff_factors(None, [0.9, 1.1])
        assert missing == [0.9, 1.1]
        assert stale == []

    def test_all_present_returns_empty(self):
        from applications.threshold_research.weight_scaling import _diff_factors

        existing = {"factors": {"0.9": {}, "1.1": {}}}
        missing, stale = _diff_factors(existing, [0.9, 1.1])
        assert missing == []
        assert stale == []

    def test_partial_overlap(self):
        from applications.threshold_research.weight_scaling import _diff_factors

        existing = {"factors": {"0.9": {}, "0.7": {}}}
        missing, stale = _diff_factors(existing, [0.9, 1.1])
        assert missing == [1.1]
        assert "0.7" in stale

    def test_all_stale(self):
        from applications.threshold_research.weight_scaling import _diff_factors

        existing = {"factors": {"0.5": {}, "0.6": {}}}
        missing, stale = _diff_factors(existing, [0.9, 1.1])
        assert missing == [0.9, 1.1]
        assert set(stale) == {"0.5", "0.6"}


class TestRunPerturbationSweep:
    def test_output_shapes_and_baseline(self, train_loader, val_loader):
        from applications.threshold_research.neuron_perturbation import (
            run_perturbation_sweep,
        )
        from applications.deep_linear.model import create_model
        from spiking import save_model

        setup = {
            "threshold_init": {
                "avg_threshold": 5.0,
                "min_threshold": 1.0,
                "std_dev": 0.5,
            },
        }
        torch.manual_seed(42)
        model = create_model(setup, NUM_INPUTS, [16])
        num_outputs = model.layers[0].num_outputs

        model_path = "/tmp/test_perturbation_model.pth"
        save_model(model, model_path)

        result = run_perturbation_sweep(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
        )

        num_fracs = len(result["perturbation_fractions"])
        assert num_fracs == 31  # -0.5 to +0.25, step 0.025

        assert len(result["original_thresholds"]) == num_outputs
        assert np.array(result["accuracy_matrix"]).shape == (num_outputs, num_fracs)
        assert np.array(result["f1_matrix"]).shape == (num_outputs, num_fracs)
        assert len(result["optimal_thresholds"]) == num_outputs
        assert len(result["optimal_deltas"]) == num_outputs
        assert "accuracy" in result["baseline"]

    def test_zero_perturbation_matches_baseline(self, train_loader, val_loader):
        """Perturbation fraction 0 should reproduce baseline accuracy."""
        from applications.threshold_research.neuron_perturbation import (
            run_perturbation_sweep,
        )
        from applications.deep_linear.model import create_model
        from spiking import save_model

        setup = {
            "threshold_init": {
                "avg_threshold": 5.0,
                "min_threshold": 1.0,
                "std_dev": 0.5,
            },
        }
        torch.manual_seed(42)
        model = create_model(setup, NUM_INPUTS, [16])

        model_path = "/tmp/test_perturbation_zero_model.pth"
        save_model(model, model_path)

        result = run_perturbation_sweep(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
        )

        fracs = result["perturbation_fractions"]
        zero_idx = fracs.index(0.0)
        baseline_acc = result["baseline"]["accuracy"]

        # Each neuron at fraction=0 should give baseline accuracy
        acc_matrix = np.array(result["accuracy_matrix"])
        for neuron_idx in range(acc_matrix.shape[0]):
            assert acc_matrix[neuron_idx, zero_idx] == pytest.approx(
                baseline_acc, abs=1e-6
            ), f"neuron {neuron_idx} at frac=0 differs from baseline"


class TestInferSpikeTimesAndPotentialsBatch:
    def test_returns_correct_shapes(self):
        layer = _make_tiny_layer(num_inputs=8, num_outputs=4)
        B = 5
        input_times = torch.rand(B, 8)
        input_times[torch.rand(B, 8) > 0.5] = float("inf")

        spike_times, cum_potential = layer.infer_spike_times_and_potentials_batch(
            input_times
        )

        assert spike_times.shape == (B, 4)
        assert cum_potential.shape == (B, 4)

    def test_spike_times_match_infer_spike_times_batch(self):
        torch.manual_seed(99)
        layer = _make_tiny_layer(num_inputs=8, num_outputs=4)
        B = 10
        input_times = torch.rand(B, 8)
        input_times[torch.rand(B, 8) > 0.5] = float("inf")

        expected = layer.infer_spike_times_batch(input_times)
        spike_times, _ = layer.infer_spike_times_and_potentials_batch(input_times)

        torch.testing.assert_close(spike_times, expected)

    def test_all_infinite_inputs(self):
        layer = _make_tiny_layer(num_inputs=8, num_outputs=4)
        input_times = torch.full((3, 8), float("inf"))

        spike_times, cum_potential = layer.infer_spike_times_and_potentials_batch(
            input_times
        )

        assert torch.isinf(spike_times).all()
        assert (cum_potential == 0).all()


class TestComputePostHocMetrics:
    def test_output_keys_and_shapes(self, train_loader, val_loader):
        from applications.threshold_research.analysis import compute_post_hoc_metrics
        from applications.deep_linear.model import create_model
        from spiking import save_model

        setup = {
            "threshold_init": {
                "avg_threshold": 5.0,
                "min_threshold": 1.0,
                "std_dev": 0.5,
            },
        }
        torch.manual_seed(42)
        num_outputs = 16
        model = create_model(setup, NUM_INPUTS, [num_outputs])

        model_path = "/tmp/test_post_hoc_model.pth"
        save_model(model, model_path)

        metrics = compute_post_hoc_metrics(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
        )

        expected_keys = {
            "weight_l2_norm",
            "weight_l1_norm",
            "avg_spike_time",
            "spike_rate",
            "weight_std",
            "potential_ratio_mean",
            "potential_ratio_max",
            "potential_ratio_std",
        }
        assert set(metrics.keys()) == expected_keys

        for key in expected_keys:
            arr = np.array(metrics[key])
            assert arr.shape == (num_outputs,), f"{key} has wrong shape: {arr.shape}"

    def test_spike_rate_bounded(self, train_loader, val_loader):
        from applications.threshold_research.analysis import compute_post_hoc_metrics
        from applications.deep_linear.model import create_model
        from spiking import save_model

        setup = {
            "threshold_init": {
                "avg_threshold": 5.0,
                "min_threshold": 1.0,
                "std_dev": 0.5,
            },
        }
        torch.manual_seed(42)
        model = create_model(setup, NUM_INPUTS, [16])

        model_path = "/tmp/test_post_hoc_rate_model.pth"
        save_model(model, model_path)

        metrics = compute_post_hoc_metrics(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
        )

        rates = np.array(metrics["spike_rate"])
        assert np.all(rates >= 0.0)
        assert np.all(rates <= 1.0)

    def test_potential_ratio_max_ge_mean(self, train_loader, val_loader):
        from applications.threshold_research.analysis import compute_post_hoc_metrics
        from applications.deep_linear.model import create_model
        from spiking import save_model

        setup = {
            "threshold_init": {
                "avg_threshold": 5.0,
                "min_threshold": 1.0,
                "std_dev": 0.5,
            },
        }
        torch.manual_seed(42)
        model = create_model(setup, NUM_INPUTS, [16])

        model_path = "/tmp/test_post_hoc_ratio_model.pth"
        save_model(model, model_path)

        metrics = compute_post_hoc_metrics(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
        )

        ratio_max = np.array(metrics["potential_ratio_max"])
        ratio_mean = np.array(metrics["potential_ratio_mean"])
        assert np.all(ratio_max >= ratio_mean - 1e-6)

    def test_potential_ratio_mean_nonnegative(self, train_loader, val_loader):
        from applications.threshold_research.analysis import compute_post_hoc_metrics
        from applications.deep_linear.model import create_model
        from spiking import save_model

        setup = {
            "threshold_init": {
                "avg_threshold": 5.0,
                "min_threshold": 1.0,
                "std_dev": 0.5,
            },
        }
        torch.manual_seed(42)
        model = create_model(setup, NUM_INPUTS, [16])

        model_path = "/tmp/test_post_hoc_ratio_nonneg_model.pth"
        save_model(model, model_path)

        metrics = compute_post_hoc_metrics(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
        )

        ratio_mean = np.array(metrics["potential_ratio_mean"])
        assert np.all(ratio_mean >= 0.0)

    def test_weight_norms_positive(self, train_loader, val_loader):
        from applications.threshold_research.analysis import compute_post_hoc_metrics
        from applications.deep_linear.model import create_model
        from spiking import save_model

        setup = {
            "threshold_init": {
                "avg_threshold": 5.0,
                "min_threshold": 1.0,
                "std_dev": 0.5,
            },
        }
        torch.manual_seed(42)
        model = create_model(setup, NUM_INPUTS, [16])

        model_path = "/tmp/test_post_hoc_norms_model.pth"
        save_model(model, model_path)

        metrics = compute_post_hoc_metrics(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
        )

        assert np.all(np.array(metrics["weight_l2_norm"]) > 0)
        assert np.all(np.array(metrics["weight_l1_norm"]) > 0)


class TestTrainWithMetrics:
    def test_output_structure(self, train_loader, val_loader):
        from applications.threshold_research.train_models import train_with_metrics

        metrics = train_with_metrics(
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
            avg_threshold=5.0,
            output_dir="/tmp/test_train_with_metrics",
            num_epochs=1,
        )

        expected_keys = {
            "winner_counts",
            "spike_counts",
            "update_counts",
            "threshold_initial",
            "threshold_final",
            "threshold_drift",
        }
        assert set(metrics.keys()) == expected_keys

        # All arrays should have same length (num_outputs)
        lengths = {len(metrics[k]) for k in expected_keys}
        assert len(lengths) == 1, f"Inconsistent lengths: {lengths}"

    def test_threshold_drift_consistency(self, train_loader, val_loader):
        from applications.threshold_research.train_models import train_with_metrics

        metrics = train_with_metrics(
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
            avg_threshold=5.0,
            output_dir="/tmp/test_train_drift",
            num_epochs=1,
        )

        initial = np.array(metrics["threshold_initial"])
        final = np.array(metrics["threshold_final"])
        drift = np.array(metrics["threshold_drift"])
        np.testing.assert_allclose(drift, final - initial, atol=1e-6)

    def test_saves_training_metrics_json(self, train_loader, val_loader):
        from applications.threshold_research.train_models import train_with_metrics

        output_dir = "/tmp/test_train_metrics_json"
        train_with_metrics(
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
            avg_threshold=5.0,
            output_dir=output_dir,
            num_epochs=1,
        )

        metrics_path = os.path.join(output_dir, "training_metrics.json")
        assert os.path.exists(metrics_path)

        with open(metrics_path) as f:
            saved = json.load(f)
        assert "winner_counts" in saved
        assert "spike_counts" in saved

    def test_counts_nonnegative(self, train_loader, val_loader):
        from applications.threshold_research.train_models import train_with_metrics

        metrics = train_with_metrics(
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
            avg_threshold=5.0,
            output_dir="/tmp/test_train_nonneg",
            num_epochs=1,
        )

        assert all(v >= 0 for v in metrics["winner_counts"])
        assert all(v >= 0 for v in metrics["spike_counts"])
        assert all(v >= 0 for v in metrics["update_counts"])


class TestCorrelationsReportWithTrainingMetrics:
    def test_new_correlations_appear(self):
        from applications.threshold_research.analysis import correlations_report

        num_neurons = 16
        rng = np.random.RandomState(42)

        results = {
            "original_thresholds": rng.uniform(3, 8, num_neurons).tolist(),
            "optimal_deltas": rng.uniform(-1, 1, num_neurons).tolist(),
            "accuracy_matrix": rng.uniform(0.1, 0.9, (num_neurons, 5)).tolist(),
        }
        training_metrics = {
            "spike_counts": rng.randint(0, 100, num_neurons).tolist(),
            "update_counts": rng.randint(0, 50, num_neurons).tolist(),
            "threshold_drift": rng.uniform(-2, 2, num_neurons).tolist(),
        }
        post_hoc_metrics = {
            "weight_l2_norm": rng.uniform(1, 10, num_neurons).tolist(),
            "weight_l1_norm": rng.uniform(5, 50, num_neurons).tolist(),
            "avg_spike_time": rng.uniform(0, 1, num_neurons).tolist(),
            "spike_rate": rng.uniform(0, 1, num_neurons).tolist(),
            "weight_std": rng.uniform(0.01, 0.5, num_neurons).tolist(),
            "potential_ratio_mean": rng.uniform(0, 1, num_neurons).tolist(),
            "potential_ratio_max": rng.uniform(0.5, 1.5, num_neurons).tolist(),
            "potential_ratio_std": rng.uniform(0, 0.5, num_neurons).tolist(),
        }

        report = correlations_report(
            results,
            training_metrics=training_metrics,
            post_hoc_metrics=post_hoc_metrics,
        )

        # Check new correlation keys exist
        for key in [
            "spike_count_vs_delta",
            "update_count_vs_delta",
            "threshold_drift_vs_delta",
            "weight_l2_vs_delta",
            "weight_l1_vs_delta",
            "avg_spike_time_vs_delta",
            "spike_rate_vs_delta",
            "weight_std_vs_delta",
            "potential_ratio_mean_vs_delta",
            "potential_ratio_max_vs_delta",
            "potential_ratio_std_vs_delta",
        ]:
            assert key in report, f"Missing correlation: {key}"
            assert "r" in report[key]
            assert "p" in report[key]

    def test_backward_compatible_without_new_metrics(self):
        """correlations_report still works without training/post-hoc metrics."""
        from applications.threshold_research.analysis import correlations_report

        num_neurons = 16
        rng = np.random.RandomState(42)

        results = {
            "original_thresholds": rng.uniform(3, 8, num_neurons).tolist(),
            "optimal_deltas": rng.uniform(-1, 1, num_neurons).tolist(),
            "accuracy_matrix": rng.uniform(0.1, 0.9, (num_neurons, 5)).tolist(),
        }

        report = correlations_report(results)
        assert "threshold_vs_delta" in report
        assert "spike_count_vs_delta" not in report


class TestComputePerturbedFeatures:
    def test_output_structure(self, train_loader, val_loader):
        from applications.threshold_research.neuron_perturbation import (
            compute_perturbed_features,
        )
        from applications.deep_linear.model import create_model
        from spiking import save_model

        setup = {
            "threshold_init": {
                "avg_threshold": 5.0,
                "min_threshold": 1.0,
                "std_dev": 0.5,
            },
        }
        torch.manual_seed(42)
        model = create_model(setup, NUM_INPUTS, [16])
        num_outputs = model.layers[0].num_outputs

        model_path = "/tmp/test_compute_features_model.pth"
        save_model(model, model_path)

        features = compute_perturbed_features(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
        )

        n_train = features["baseline_train"].shape[0]
        n_val = features["baseline_val"].shape[0]
        assert features["baseline_train"].shape == (n_train, num_outputs)
        assert features["baseline_val"].shape == (n_val, num_outputs)
        assert features["perturbed_train"].shape == (31, n_train, num_outputs)
        assert features["perturbed_val"].shape == (31, n_val, num_outputs)
        assert features["labels_train"].shape == (n_train,)
        assert features["labels_val"].shape == (n_val,)
        assert len(features["original_thresholds"]) == num_outputs
        assert len(features["perturbation_fractions"]) == 31

    def test_caching_roundtrip(self, train_loader, val_loader, tmp_path):
        from applications.threshold_research.neuron_perturbation import (
            compute_perturbed_features,
        )
        from applications.deep_linear.model import create_model
        from spiking import save_model

        setup = {
            "threshold_init": {
                "avg_threshold": 5.0,
                "min_threshold": 1.0,
                "std_dev": 0.5,
            },
        }
        torch.manual_seed(42)
        model = create_model(setup, NUM_INPUTS, [16])

        model_path = str(tmp_path / "model.pth")
        save_model(model, model_path)
        cache_dir = str(tmp_path / "cache")

        # First call computes and caches
        features1 = compute_perturbed_features(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
            cache_dir=cache_dir,
        )

        # Verify cache files exist
        assert os.path.exists(os.path.join(cache_dir, "metadata.json"))
        assert os.path.exists(os.path.join(cache_dir, "baseline_train.npy"))
        assert os.path.exists(os.path.join(cache_dir, "perturbed_train.npy"))

        # Second call loads from cache
        features2 = compute_perturbed_features(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
            cache_dir=cache_dir,
        )

        np.testing.assert_array_equal(
            features1["baseline_train"], features2["baseline_train"]
        )
        np.testing.assert_array_equal(
            features1["perturbed_train"], features2["perturbed_train"]
        )
        assert features1["original_thresholds"] == features2["original_thresholds"]


class TestEvaluatePerturbationsResume:
    def test_resume_produces_correct_results(self, train_loader, val_loader, tmp_path):
        """Simulate interruption by writing partial results, then resuming."""
        from applications.threshold_research.neuron_perturbation import (
            compute_perturbed_features,
            evaluate_perturbations,
        )
        from applications.deep_linear.model import create_model
        from spiking import save_model

        setup = {
            "threshold_init": {
                "avg_threshold": 5.0,
                "min_threshold": 1.0,
                "std_dev": 0.5,
            },
        }
        torch.manual_seed(42)
        model = create_model(setup, NUM_INPUTS, [16])

        model_path = str(tmp_path / "model.pth")
        save_model(model, model_path)

        features = compute_perturbed_features(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
        )

        # Full run without caching (reference)
        full_result = evaluate_perturbations(features=features)

        # Simulate partial run: save first 2 fractions as completed
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir, exist_ok=True)
        acc_matrix = np.array(full_result["accuracy_matrix"])
        f1_matrix_arr = np.array(full_result["f1_matrix"])
        partial = {
            "completed_fractions": [0, 1],
            "accuracy_matrix": acc_matrix[:, [0, 1]].tolist(),
            "f1_matrix": f1_matrix_arr[:, [0, 1]].tolist(),
        }
        with open(os.path.join(cache_dir, "partial_results.json"), "w") as f:
            json.dump(partial, f)

        # Resume
        resumed_result = evaluate_perturbations(
            features=features, cache_dir=cache_dir
        )

        np.testing.assert_array_almost_equal(
            resumed_result["accuracy_matrix"], full_result["accuracy_matrix"]
        )
        np.testing.assert_array_almost_equal(
            resumed_result["f1_matrix"], full_result["f1_matrix"]
        )


class TestCachedPerturbationSweep:
    def test_cached_matches_uncached(self, train_loader, val_loader, tmp_path):
        from applications.threshold_research.neuron_perturbation import (
            run_perturbation_sweep,
        )
        from applications.deep_linear.model import create_model
        from spiking import save_model

        setup = {
            "threshold_init": {
                "avg_threshold": 5.0,
                "min_threshold": 1.0,
                "std_dev": 0.5,
            },
        }
        torch.manual_seed(42)
        model = create_model(setup, NUM_INPUTS, [16])

        model_path = str(tmp_path / "model.pth")
        save_model(model, model_path)

        # Uncached run
        uncached = run_perturbation_sweep(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
        )

        # Cached run
        cache_dir = str(tmp_path / "cache")
        cached = run_perturbation_sweep(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
            cache_dir=cache_dir,
        )

        assert uncached["baseline"] == cached["baseline"]
        np.testing.assert_array_almost_equal(
            uncached["accuracy_matrix"], cached["accuracy_matrix"]
        )
        np.testing.assert_array_almost_equal(
            uncached["f1_matrix"], cached["f1_matrix"]
        )
        assert uncached["original_thresholds"] == pytest.approx(
            cached["original_thresholds"]
        )
        assert uncached["optimal_thresholds"] == pytest.approx(
            cached["optimal_thresholds"]
        )


class TestComputePredictiveModel:
    def test_returns_r_squared(self):
        from applications.threshold_research.analysis import compute_predictive_model

        num_neurons = 32
        rng = np.random.RandomState(42)

        optimal_deltas = rng.uniform(-1, 1, num_neurons)
        metrics = {
            "weight_l2_norm": rng.uniform(1, 10, num_neurons),
            "spike_rate": rng.uniform(0, 1, num_neurons),
            "threshold_drift": rng.uniform(-2, 2, num_neurons),
        }

        result = compute_predictive_model(optimal_deltas, metrics)
        assert "r_squared" in result
        assert "coefficients" in result
        assert 0.0 <= result["r_squared"] <= 1.0
        assert len(result["coefficients"]) == len(metrics)

    def test_perfect_predictor(self):
        from applications.threshold_research.analysis import compute_predictive_model

        # If optimal_delta = 2*x + 3*y, the model should find R² ≈ 1.0
        num_neurons = 50
        rng = np.random.RandomState(42)
        x = rng.uniform(0, 1, num_neurons)
        y = rng.uniform(0, 1, num_neurons)
        optimal_deltas = 2 * x + 3 * y

        metrics = {"x": x, "y": y}
        result = compute_predictive_model(optimal_deltas, metrics)
        assert result["r_squared"] > 0.99
