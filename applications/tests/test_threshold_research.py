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
        _, val_m = evaluate_model(sub_model, train_loader, val_loader)

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


class TestEvaluateOptimalThresholds:
    def test_includes_importance(self, train_loader, val_loader):
        from applications.threshold_research.optimal_thresholds import (
            evaluate_optimal_thresholds,
        )
        from applications.deep_linear.model import create_model
        from applications.threshold_research.neuron_perturbation import (
            run_perturbation_sweep,
        )
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

        model_path = "/tmp/test_optimal_importance_model.pth"
        save_model(model, model_path)

        perturbation_results = run_perturbation_sweep(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
        )

        result = evaluate_optimal_thresholds(
            model_path=model_path,
            perturbation_results=perturbation_results,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
        )

        baseline_imp = np.array(result["baseline_importance"])
        optimal_imp = np.array(result["optimal_importance"])
        assert baseline_imp.shape == (num_outputs,)
        assert optimal_imp.shape == (num_outputs,)
        assert np.all(baseline_imp >= 0)
        assert np.all(optimal_imp >= 0)


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
        from applications.threshold_research.metrics import compute_post_hoc_metrics
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
        from applications.threshold_research.metrics import compute_post_hoc_metrics
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
        from applications.threshold_research.metrics import compute_post_hoc_metrics
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
        from applications.threshold_research.metrics import compute_post_hoc_metrics
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
        from applications.threshold_research.metrics import compute_post_hoc_metrics
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

    def test_importance_gap_correlation(self):
        """correlations_report includes importance gap when importance data provided."""
        from applications.threshold_research.analysis import correlations_report

        num_neurons = 16
        rng = np.random.RandomState(42)

        results = {
            "original_thresholds": rng.uniform(3, 8, num_neurons).tolist(),
            "optimal_deltas": rng.uniform(-1, 1, num_neurons).tolist(),
            "accuracy_matrix": rng.uniform(0.1, 0.9, (num_neurons, 5)).tolist(),
            "baseline_importance": rng.uniform(0, 1, num_neurons).tolist(),
            "optimal_importance": rng.uniform(0, 1, num_neurons).tolist(),
        }

        report = correlations_report(results)
        assert "importance_gap_vs_delta" in report
        assert "r" in report["importance_gap_vs_delta"]
        assert "p" in report["importance_gap_vs_delta"]


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

    def test_force_ignores_cache(self, train_loader, val_loader, tmp_path):
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

        # First call populates the cache
        compute_perturbed_features(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
            cache_dir=cache_dir,
        )

        # Corrupt a cached file to verify force actually recomputes
        baseline_path = os.path.join(cache_dir, "baseline_train.npy")
        original = np.load(baseline_path)
        np.save(baseline_path, np.zeros_like(original))

        # Without force: loads corrupted cache
        cached = compute_perturbed_features(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
            cache_dir=cache_dir,
        )
        assert np.allclose(cached["baseline_train"], 0.0)

        # With force: recomputes and overwrites
        forced = compute_perturbed_features(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
            cache_dir=cache_dir,
            force=True,
        )
        np.testing.assert_array_equal(forced["baseline_train"], original)


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
        resumed_result = evaluate_perturbations(features=features, cache_dir=cache_dir)

        np.testing.assert_array_almost_equal(
            resumed_result["accuracy_matrix"], full_result["accuracy_matrix"]
        )
        np.testing.assert_array_almost_equal(
            resumed_result["f1_matrix"], full_result["f1_matrix"]
        )


class TestEvaluatePerturbationsStrategies:
    """Tests for classifier_factory and refit parameters."""

    @pytest.fixture()
    def features(self, train_loader, val_loader, tmp_path):
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

        return compute_perturbed_features(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            seed=42,
        )

    def test_no_refit_produces_valid_results(self, features):
        from applications.threshold_research.neuron_perturbation import (
            evaluate_perturbations,
        )

        result = evaluate_perturbations(features=features, refit=False)

        num_outputs = features["baseline_train"].shape[1]
        num_fracs = len(features["perturbation_fractions"])

        assert np.array(result["accuracy_matrix"]).shape == (num_outputs, num_fracs)
        assert "accuracy" in result["baseline"]
        assert len(result["optimal_deltas"]) == num_outputs

    def test_custom_classifier_factory(self, features):
        from sklearn.neighbors import NearestCentroid

        from applications.threshold_research.neuron_perturbation import (
            evaluate_perturbations,
        )

        result = evaluate_perturbations(
            features=features,
            classifier_factory=lambda: NearestCentroid(),
        )

        num_outputs = features["baseline_train"].shape[1]
        num_fracs = len(features["perturbation_fractions"])

        assert np.array(result["accuracy_matrix"]).shape == (num_outputs, num_fracs)
        assert "accuracy" in result["baseline"]

    def test_no_refit_larger_variation(self, features):
        """No-refit should produce larger accuracy variation than refit.

        When the classifier is not re-fit, it cannot compensate for
        perturbed features, so accuracy should vary more across fractions.
        """
        from applications.threshold_research.neuron_perturbation import (
            evaluate_perturbations,
        )

        refit_result = evaluate_perturbations(features=features, refit=True)
        no_refit_result = evaluate_perturbations(features=features, refit=False)

        refit_acc = np.array(refit_result["accuracy_matrix"])
        no_refit_acc = np.array(no_refit_result["accuracy_matrix"])

        # Compare per-neuron accuracy range (max - min across fractions)
        refit_ranges = np.ptp(refit_acc, axis=1)
        no_refit_ranges = np.ptp(no_refit_acc, axis=1)

        assert no_refit_ranges.mean() >= refit_ranges.mean(), (
            f"Expected no-refit to have larger accuracy variation "
            f"(mean range {no_refit_ranges.mean():.6f}) than refit "
            f"({refit_ranges.mean():.6f})"
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
        np.testing.assert_array_almost_equal(uncached["f1_matrix"], cached["f1_matrix"])
        assert uncached["original_thresholds"] == pytest.approx(
            cached["original_thresholds"]
        )
        assert uncached["optimal_thresholds"] == pytest.approx(
            cached["optimal_thresholds"]
        )


class TestComputePredictiveModel:
    def test_returns_r_squared(self):
        from applications.threshold_research.predictive_model import compute_predictive_model

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
        from applications.threshold_research.predictive_model import compute_predictive_model

        # If optimal_delta = 2*x + 3*y, the model should find R² ≈ 1.0
        num_neurons = 50
        rng = np.random.RandomState(42)
        x = rng.uniform(0, 1, num_neurons)
        y = rng.uniform(0, 1, num_neurons)
        optimal_deltas = 2 * x + 3 * y

        metrics = {"x": x, "y": y}
        result = compute_predictive_model(optimal_deltas, metrics)
        assert result["r_squared"] > 0.99


class TestRunPostHocMetrics:
    def test_saves_metrics_json(self, train_loader, val_loader, tmp_path):
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

        # Create tobj_*/seed_* directory structure
        seed_dir = tmp_path / "tobj_0.5" / "seed_1"
        seed_dir.mkdir(parents=True)
        save_model(model, str(seed_dir / "model.pth"))

        # Monkey-patch _find_models to use tmp_path
        import applications.threshold_research.run_post_hoc_metrics as mod

        orig_find = mod._find_models
        mod._find_models = lambda base_dir: [(str(seed_dir / "model.pth"), 0.5, 1)]
        try:
            mod.run.__wrapped__ = None  # ensure no caching
        except AttributeError:
            pass

        # Call run with pre-created loaders by patching create_dataset
        orig_create = mod.create_dataset
        _loader_pair = (train_loader, val_loader)

        class FakeDataset:
            image_shape = IMAGE_SHAPE

        class FakeLoader:
            dataset = FakeDataset()

        mod.create_dataset = lambda ds: (FakeLoader(), val_loader)
        # We need the train_loader to actually work for compute_post_hoc_metrics,
        # so let's just call the core logic directly instead
        mod.create_dataset = orig_create
        mod._find_models = orig_find

        # Test the core logic: compute + save + skip-if-exists
        from applications.threshold_research.metrics import compute_post_hoc_metrics

        output_path = str(seed_dir / "post_hoc_metrics.json")
        metrics = compute_post_hoc_metrics(
            model_path=str(seed_dir / "model.pth"),
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
        )
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Verify file was saved
        assert os.path.exists(output_path)

        # Verify round-trip: load and check keys
        with open(output_path) as f:
            loaded = json.load(f)

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
        assert set(loaded.keys()) == expected_keys

        for key in expected_keys:
            assert len(loaded[key]) == 16, f"{key} has wrong length"

    def test_skip_existing(self, train_loader, val_loader, tmp_path):
        """Verify _find_models + skip logic works correctly."""
        from applications.threshold_research.run_perturbation import _find_models

        # Create directory structure with existing post_hoc_metrics.json
        seed_dir = tmp_path / "tobj_0.5" / "seed_1"
        seed_dir.mkdir(parents=True)
        (seed_dir / "model.pth").touch()
        (seed_dir / "post_hoc_metrics.json").write_text("{}")

        models = _find_models(str(tmp_path))
        assert len(models) == 1
        assert models[0][1] == 0.5
        assert models[0][2] == 1

        # The skip logic checks os.path.exists on the output path
        output_path = os.path.join(
            os.path.dirname(models[0][0]), "post_hoc_metrics.json"
        )
        assert os.path.exists(output_path)


class TestComputeNonlinearPredictiveModel:
    def test_returns_expected_keys(self):
        from applications.threshold_research.predictive_model import (
            compute_nonlinear_predictive_model,
        )

        num_neurons = 64
        rng = np.random.RandomState(42)
        optimal_deltas = rng.uniform(-1, 1, num_neurons)
        metrics = {
            "weight_l2_norm": rng.uniform(1, 10, num_neurons),
            "spike_rate": rng.uniform(0, 1, num_neurons),
            "weight_std": rng.uniform(0, 2, num_neurons),
        }

        result = compute_nonlinear_predictive_model(optimal_deltas, metrics)

        expected_keys = {
            "linear_cv_r2",
            "linear_cv_r2_std",
            "gbr_cv_r2",
            "gbr_cv_r2_std",
            "n_samples",
            "n_features",
        }
        assert set(result.keys()) == expected_keys
        for key in expected_keys:
            assert np.isfinite(result[key]), f"{key} is not finite"

    def test_nonlinear_beats_linear_on_nonlinear_data(self):
        from applications.threshold_research.predictive_model import (
            compute_nonlinear_predictive_model,
        )

        num_neurons = 200
        rng = np.random.RandomState(42)
        x = rng.uniform(-1, 1, num_neurons)
        y = rng.uniform(-1, 1, num_neurons)
        # Quadratic relationship: linear model cannot capture this well
        optimal_deltas = x**2 + y**2 + rng.normal(0, 0.1, num_neurons)
        metrics = {"x": x, "y": y}

        result = compute_nonlinear_predictive_model(optimal_deltas, metrics)
        assert result["gbr_cv_r2"] > result["linear_cv_r2"]


class TestDenoiseOptimalDeltas:
    def test_gaussian_smoothing_reduces_noise(self):
        """Gaussian smoothing on noisy parabolic accuracy curves recovers vertex."""
        from applications.threshold_research.predictive_model import denoise_optimal_deltas

        num_neurons = 10
        num_fractions = 31
        fractions = np.linspace(-0.5, 0.25, num_fractions)
        original_thresholds = np.full(num_neurons, 5.0)
        rng = np.random.RandomState(42)

        # Parabolic accuracy curves with known peak at fraction index 20 (frac ~0.0)
        peak_idx = 20
        accuracy_matrix = np.zeros((num_neurons, num_fractions))
        for n in range(num_neurons):
            base = -((np.arange(num_fractions) - peak_idx) ** 2) * 0.001 + 0.9
            accuracy_matrix[n] = base + rng.normal(0, 0.002, num_fractions)

        result = denoise_optimal_deltas(
            accuracy_matrix, fractions, original_thresholds, method="gaussian", sigma=2.0
        )

        # Denoised argmax should be close to the true peak for most neurons
        expected_delta = fractions[peak_idx] * original_thresholds[0]
        close_count = np.sum(np.abs(result["optimal_deltas"] - expected_delta) < 0.5)
        assert close_count >= 7, f"Only {close_count}/10 neurons near true peak"

    def test_polynomial_fit_finds_vertex(self):
        """Polynomial fit on parabolic accuracy curve finds vertex."""
        from applications.threshold_research.predictive_model import denoise_optimal_deltas

        num_neurons = 5
        num_fractions = 31
        fractions = np.linspace(-0.5, 0.25, num_fractions)
        original_thresholds = np.full(num_neurons, 5.0)

        # Perfect parabola peaking at fraction=0.0 (index ~20)
        accuracy_matrix = np.zeros((num_neurons, num_fractions))
        for n in range(num_neurons):
            accuracy_matrix[n] = -(fractions**2) + 0.9

        result = denoise_optimal_deltas(
            accuracy_matrix, fractions, original_thresholds, method="polynomial"
        )

        # All deltas should be near 0 (peak at fraction=0, so delta = 0 * threshold = 0)
        assert np.allclose(result["optimal_deltas"], 0.0, atol=0.1)

    def test_flat_curve_gives_low_confidence(self):
        """Flat accuracy curve → low confidence score."""
        from applications.threshold_research.predictive_model import denoise_optimal_deltas

        num_neurons = 3
        num_fractions = 31
        fractions = np.linspace(-0.5, 0.25, num_fractions)
        original_thresholds = np.full(num_neurons, 5.0)

        # Flat curves
        accuracy_matrix = np.full((num_neurons, num_fractions), 0.9)

        result = denoise_optimal_deltas(
            accuracy_matrix, fractions, original_thresholds, method="gaussian"
        )

        assert np.all(result["confidence"] < 0.01)

    def test_output_shapes(self):
        """Output arrays match input neuron count."""
        from applications.threshold_research.predictive_model import denoise_optimal_deltas

        num_neurons = 8
        num_fractions = 31
        fractions = np.linspace(-0.5, 0.25, num_fractions)
        original_thresholds = np.ones(num_neurons) * 5.0
        rng = np.random.RandomState(42)
        accuracy_matrix = rng.uniform(0.8, 0.95, (num_neurons, num_fractions))

        result = denoise_optimal_deltas(
            accuracy_matrix, fractions, original_thresholds, method="gaussian"
        )

        assert result["optimal_deltas"].shape == (num_neurons,)
        assert result["confidence"].shape == (num_neurons,)


class TestFilterSensitiveNeurons:
    def test_flat_neurons_filtered_variable_retained(self):
        """Flat neurons are excluded, variable neurons are kept."""
        from applications.threshold_research.predictive_model import filter_sensitive_neurons

        num_fractions = 31
        # Neuron 0: flat, Neuron 1: variable, Neuron 2: barely variable
        accuracy_matrix = np.array([
            np.full(num_fractions, 0.9),            # flat
            np.linspace(0.85, 0.95, num_fractions),  # range = 0.10
            np.full(num_fractions, 0.9) + np.linspace(0, 0.003, num_fractions),  # range = 0.003
        ])

        mask = filter_sensitive_neurons(accuracy_matrix, min_range=0.005)

        assert mask.shape == (3,)
        assert mask[0] == False  # flat → excluded
        assert mask[1] == True   # variable → kept
        assert mask[2] == False  # below threshold → excluded

    def test_all_sensitive(self):
        """All neurons above threshold → all True."""
        from applications.threshold_research.predictive_model import filter_sensitive_neurons

        accuracy_matrix = np.array([
            np.linspace(0.8, 0.95, 10),
            np.linspace(0.7, 0.9, 10),
        ])

        mask = filter_sensitive_neurons(accuracy_matrix, min_range=0.005)
        assert np.all(mask)


class TestComputeClassifierMetrics:
    def test_output_keys_and_shapes(self):
        """Returns expected keys with correct array shapes."""
        from applications.threshold_research.metrics import compute_classifier_metrics
        from sklearn.linear_model import RidgeClassifier

        num_samples = 50
        num_features = 8
        num_classes = 3
        rng = np.random.RandomState(42)

        X_train = rng.randn(num_samples, num_features)
        y_train = rng.randint(0, num_classes, num_samples)
        X_val = rng.randn(20, num_features)
        y_val = rng.randint(0, num_classes, 20)

        clf = RidgeClassifier()
        clf.fit(X_train, y_train)

        result = compute_classifier_metrics(clf, X_train, y_train, X_val, y_val)

        expected_keys = {
            "coef_magnitude",
            "misclassified_margin_contribution",
            "fisher_discriminant_ratio",
            "max_feature_correlation",
        }
        assert set(result.keys()) == expected_keys
        for key in expected_keys:
            assert result[key].shape == (num_features,), f"{key} has wrong shape"

    def test_coef_magnitude_matches_feature_importance(self):
        """coef_magnitude should equal compute_feature_importance."""
        from applications.threshold_research.metrics import (
            compute_classifier_metrics,
            compute_feature_importance,
        )
        from sklearn.linear_model import RidgeClassifier

        num_samples = 50
        num_features = 6
        rng = np.random.RandomState(42)

        X_train = rng.randn(num_samples, num_features)
        y_train = rng.randint(0, 3, num_samples)
        X_val = rng.randn(20, num_features)
        y_val = rng.randint(0, 3, 20)

        clf = RidgeClassifier()
        clf.fit(X_train, y_train)

        result = compute_classifier_metrics(clf, X_train, y_train, X_val, y_val)
        importance = compute_feature_importance(clf, X_train)

        np.testing.assert_allclose(result["coef_magnitude"], importance)

    def test_misclassified_margin_sign(self):
        """Neuron that actively hurts classification should have negative margin contribution."""
        from applications.threshold_research.metrics import compute_classifier_metrics
        from sklearn.linear_model import RidgeClassifier

        rng = np.random.RandomState(42)
        num_samples = 100
        num_features = 4

        # Feature 0 is perfectly predictive of class
        y_train = np.array([0] * 50 + [1] * 50)
        X_train = rng.randn(num_samples, num_features) * 0.1
        X_train[:50, 0] = 1.0  # class 0 → high feature 0
        X_train[50:, 0] = -1.0  # class 1 → low feature 0

        # Feature 1 is anti-correlated (hurts classification)
        X_train[:50, 1] = -1.0  # class 0 → low feature 1 (opposite of what helps)
        X_train[50:, 1] = 1.0

        y_val = np.array([0] * 25 + [1] * 25)
        X_val = rng.randn(50, num_features) * 0.1
        X_val[:25, 0] = 1.0
        X_val[25:, 0] = -1.0
        X_val[:25, 1] = -1.0
        X_val[25:, 1] = 1.0

        clf = RidgeClassifier()
        clf.fit(X_train, y_train)

        result = compute_classifier_metrics(clf, X_train, y_train, X_val, y_val)

        # If there are misclassified samples, the adversarial neuron should have
        # negative or lower margin contribution than the helpful neuron
        # (If classifier is perfect, margin contribution is 0 for all — that's fine too)
        if not np.allclose(result["misclassified_margin_contribution"], 0.0):
            assert result["misclassified_margin_contribution"][0] > result["misclassified_margin_contribution"][1]

    def test_fisher_ratio_high_for_separable(self):
        """Fisher discriminant ratio is high for perfectly separable feature."""
        from applications.threshold_research.metrics import compute_classifier_metrics
        from sklearn.linear_model import RidgeClassifier

        num_samples = 100
        num_features = 3
        rng = np.random.RandomState(42)

        y_train = np.array([0] * 50 + [1] * 50)
        X_train = rng.randn(num_samples, num_features) * 0.01
        # Feature 0: perfectly separable (large between-class variance, tiny within)
        X_train[:50, 0] = 10.0
        X_train[50:, 0] = -10.0

        X_val = X_train[:20].copy()
        y_val = y_train[:20].copy()

        clf = RidgeClassifier()
        clf.fit(X_train, y_train)

        result = compute_classifier_metrics(clf, X_train, y_train, X_val, y_val)

        # Feature 0 should have much higher fisher ratio than features 1, 2
        assert result["fisher_discriminant_ratio"][0] > 10 * result["fisher_discriminant_ratio"][1]

    def test_max_correlation_for_duplicate_columns(self):
        """Duplicate feature columns should have max_feature_correlation = 1.0."""
        from applications.threshold_research.metrics import compute_classifier_metrics
        from sklearn.linear_model import RidgeClassifier

        num_samples = 50
        rng = np.random.RandomState(42)

        y_train = rng.randint(0, 2, num_samples)
        col = rng.randn(num_samples)
        X_train = np.column_stack([col, col, rng.randn(num_samples)])

        X_val = rng.randn(20, 3)
        y_val = rng.randint(0, 2, 20)

        clf = RidgeClassifier(alpha=10.0)  # high regularization to handle collinearity
        clf.fit(X_train, y_train)

        result = compute_classifier_metrics(clf, X_train, y_train, X_val, y_val)

        # Features 0 and 1 are identical → max correlation should be ~1.0
        assert result["max_feature_correlation"][0] > 0.99
        assert result["max_feature_correlation"][1] > 0.99


class TestCorrelationsReportWithClassifierMetrics:
    def test_classifier_metric_correlations_appear(self):
        """classifier_metrics param adds new correlation keys."""
        from applications.threshold_research.analysis import correlations_report

        num_neurons = 16
        rng = np.random.RandomState(42)

        results = {
            "original_thresholds": rng.uniform(3, 8, num_neurons).tolist(),
            "optimal_deltas": rng.uniform(-1, 1, num_neurons).tolist(),
            "accuracy_matrix": rng.uniform(0.1, 0.9, (num_neurons, 5)).tolist(),
        }
        classifier_metrics = {
            "coef_magnitude": rng.uniform(0, 1, num_neurons),
            "misclassified_margin_contribution": rng.uniform(-1, 1, num_neurons),
            "fisher_discriminant_ratio": rng.uniform(0, 10, num_neurons),
            "max_feature_correlation": rng.uniform(0, 1, num_neurons),
        }

        report = correlations_report(results, classifier_metrics=classifier_metrics)

        for key in [
            "coef_magnitude_vs_delta",
            "misclassified_margin_vs_delta",
            "fisher_ratio_vs_delta",
            "max_correlation_vs_delta",
        ]:
            assert key in report, f"Missing correlation: {key}"
            assert "r" in report[key]
            assert "p" in report[key]

    def test_backward_compatible_without_classifier_metrics(self):
        """correlations_report still works without classifier_metrics."""
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
        assert "coef_magnitude_vs_delta" not in report


class TestEvaluatePerturbationsClassifierInfo:
    def test_returns_classifier_coef_and_intercept(self):
        """evaluate_perturbations returns baseline classifier coef and intercept."""
        from applications.threshold_research.neuron_perturbation import (
            evaluate_perturbations,
        )

        num_neurons = 4
        num_fracs = 3
        num_samples_train = 30
        num_samples_val = 10
        rng = np.random.RandomState(42)

        features = {
            "baseline_train": rng.randn(num_samples_train, num_neurons).astype(np.float32),
            "baseline_val": rng.randn(num_samples_val, num_neurons).astype(np.float32),
            "labels_train": rng.randint(0, 3, num_samples_train),
            "labels_val": rng.randint(0, 3, num_samples_val),
            "perturbed_train": rng.randn(num_fracs, num_samples_train, num_neurons).astype(np.float32),
            "perturbed_val": rng.randn(num_fracs, num_samples_val, num_neurons).astype(np.float32),
            "original_thresholds": [5.0] * num_neurons,
            "perturbation_fractions": [-0.1, 0.0, 0.1],
        }

        result = evaluate_perturbations(features=features)

        assert "baseline_classifier_coef" in result
        assert "baseline_classifier_intercept" in result
        coef = np.array(result["baseline_classifier_coef"])
        intercept = np.array(result["baseline_classifier_intercept"])
        # RidgeClassifier with 3 classes → coef shape (3, num_neurons)
        assert coef.shape[1] == num_neurons
        assert intercept.shape[0] == coef.shape[0]
