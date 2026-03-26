import torch
import pytest
import numpy as np
from torch.utils.data import DataLoader

from applications.datasets import MnistSubsetDataset
from applications.threshold_research.threshold_sync import compute_fisher_thresholds
from applications.threshold_research.neuron_perturbation import (
    precompute_cumulative_potentials,
    spike_times_from_potentials,
)

INF = float("inf")

IMAGE_SHAPE = (8, 8)
SPIKE_SHAPE = (2, *IMAGE_SHAPE)
NUM_INPUTS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * 2


@pytest.fixture(scope="module")
def train_loader():
    dataset = MnistSubsetDataset("data/mnist-subset", "train", image_shape=IMAGE_SHAPE)
    return DataLoader(dataset, batch_size=None, shuffle=True)


@pytest.fixture(scope="module")
def val_loader():
    dataset = MnistSubsetDataset("data/mnist-subset", "test", image_shape=IMAGE_SHAPE)
    return DataLoader(dataset, batch_size=None, shuffle=False)


class TestComputeFisherThresholds:
    def test_picks_threshold_improving_accuracy(self):
        """Lower threshold makes a misclassified sample spike earlier, improving accuracy.

        At threshold 3.0: c0 features [0.8, 0.8, 0.2], c1 features [0.5, 0.5, 0.5].
        SVM boundary ~0.65 → sample 2 (c0, feature 0.2) misclassified → 5/6 accuracy.
        At threshold 2.4 (frac=-0.2): sample 2 spikes at 0.6 → feature 0.8 → 6/6 accuracy.
        """
        boundary_times = torch.tensor([0.2, 0.4, 0.6, 0.75, 0.9])
        cum_potentials = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 3.5, 4.0]],  # c0: spikes at 0.6 (thresh 3.0)
                [[1.0, 2.0, 3.0, 3.5, 4.0]],  # c0: spikes at 0.6
                [
                    [0.5, 1.0, 2.5, 2.8, 3.5]
                ],  # c0: spikes at 0.9 (thresh 3.0), 0.6 (thresh ≤2.5)
                [[0.2, 0.5, 1.0, 3.0, 3.5]],  # c1: spikes at 0.75
                [[0.2, 0.5, 1.0, 3.0, 3.5]],  # c1: spikes at 0.75
                [[0.2, 0.5, 1.0, 3.0, 3.5]],  # c1: spikes at 0.75
            ]
        )
        labels = np.array([0, 0, 0, 1, 1, 1])
        original_thresholds = torch.tensor([3.0])

        result = compute_fisher_thresholds(
            cum_potentials,
            boundary_times,
            original_thresholds,
            labels,
            t_target=0.5,
            n_fractions=11,
            frac_min=-0.5,
            frac_max=0.0,
        )

        # Should pick a fraction < 0, giving a lower threshold
        assert result[0].item() < original_thresholds[0].item()

    def test_already_optimal_unchanged(self):
        """When baseline accuracy is already maximal, threshold stays the same."""
        boundary_times = torch.tensor([0.2, 0.6, 0.9])
        # c0 spikes at 0.6 (feature 0.8), c1 spikes at 0.9 (feature 0.2)
        # Clearly separable → 100% baseline accuracy → nothing to improve
        cum_potentials = torch.tensor(
            [
                [[1.0, 3.0, 4.0]],
                [[1.0, 3.0, 4.0]],
                [[0.2, 0.5, 3.0]],
                [[0.2, 0.5, 3.0]],
            ]
        )
        labels = np.array([0, 0, 1, 1])
        original_thresholds = torch.tensor([3.0])

        result = compute_fisher_thresholds(
            cum_potentials,
            boundary_times,
            original_thresholds,
            labels,
            t_target=0.5,
            n_fractions=5,
            frac_min=-0.25,
            frac_max=0.25,
        )

        assert result[0].item() == 3.0

    def test_nonspiking_neuron_keeps_original(self):
        """If neuron never fires at any fraction, threshold stays unchanged."""
        boundary_times = torch.tensor([0.5, 0.8])
        # Max cumulative is 2.0. Threshold 100 → even at frac_min=-0.5, threshold=50 >> 2.0
        cum_potentials = torch.tensor(
            [
                [[1.0, 2.0]],
                [[1.0, 2.0]],
                [[0.5, 1.5]],
                [[0.5, 1.5]],
            ]
        )
        labels = np.array([0, 0, 1, 1])
        original_thresholds = torch.tensor([100.0])

        result = compute_fisher_thresholds(
            cum_potentials,
            boundary_times,
            original_thresholds,
            labels,
            t_target=0.5,
            n_fractions=5,
            frac_min=-0.5,
            frac_max=0.25,
        )
        assert result[0].item() == 100.0

    def test_constant_feature_keeps_original(self):
        """All samples produce identical feature → accuracy unchanged → keep original."""
        # All samples have same input pattern → same spike time → same feature
        input_times = torch.tensor(
            [
                [0.1, 0.3, 0.5],
                [0.1, 0.3, 0.5],
                [0.1, 0.3, 0.5],
                [0.1, 0.3, 0.5],
            ]
        )
        labels = np.array([0, 0, 1, 1])
        weights = torch.ones(1, 3)
        cum_pot, boundary = precompute_cumulative_potentials(input_times, weights)
        original_thresholds = torch.tensor([2.0])

        result = compute_fisher_thresholds(
            cum_pot,
            boundary,
            original_thresholds,
            labels,
            t_target=0.5,
            n_fractions=5,
            frac_min=-0.25,
            frac_max=0.25,
        )
        # Accuracy is the same everywhere → keep original
        assert result[0].item() == 2.0

    def test_neurons_optimized_independently(self):
        """Two neurons with different optimal fractions each get their own optimum."""
        # Neuron 0 benefits from lower threshold, neuron 1 from higher
        input_times_c0 = torch.tensor(
            [
                [0.1, 0.2, 0.5, 0.8],
                [0.1, 0.2, 0.5, 0.8],
            ]
        )
        input_times_c1 = torch.tensor(
            [
                [0.3, 0.5, 0.7, 0.9],
                [0.3, 0.5, 0.7, 0.9],
            ]
        )
        input_times = torch.cat([input_times_c0, input_times_c1], dim=0)
        labels = np.array([0, 0, 1, 1])

        # Two neurons with different weight patterns
        weights = torch.tensor(
            [
                [2.0, 1.0, 0.5, 0.5],  # neuron 0: strong early weights
                [0.5, 0.5, 1.0, 2.0],  # neuron 1: strong late weights
            ]
        )
        cum_pot, boundary = precompute_cumulative_potentials(input_times, weights)
        original_thresholds = torch.tensor([3.0, 3.0])

        result = compute_fisher_thresholds(
            cum_pot,
            boundary,
            original_thresholds,
            labels,
            t_target=0.5,
            n_fractions=11,
            frac_min=-0.5,
            frac_max=0.5,
        )

        # Both neurons should have valid thresholds
        assert result.shape == (2,)
        assert torch.all(torch.isfinite(result))
        assert torch.all(result > 0)


class TestEvaluateFisherThresholds:
    def test_output_structure(self, train_loader, val_loader):
        """Result dict has baseline and fisher entries."""
        from applications.deep_linear.model import create_model
        from applications.threshold_research.run_sync_evaluation import (
            evaluate_fisher_thresholds,
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
        model_path = "/tmp/test_fisher_eval_model.pth"
        save_model(model, model_path)

        result = evaluate_fisher_thresholds(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            t_target=0.875,
            seed=42,
        )

        assert "baseline" in result
        assert "fisher" in result
        assert "accuracy" in result["baseline"]
        assert "f1" in result["baseline"]

        fisher = result["fisher"]
        assert "n_adjusted" in fisher
        assert "accuracy" in fisher
        assert isinstance(fisher["n_adjusted"], int)

    def test_baseline_accuracy_bounded(self, train_loader, val_loader):
        """Baseline accuracy is between 0 and 1."""
        from applications.deep_linear.model import create_model
        from applications.threshold_research.run_sync_evaluation import (
            evaluate_fisher_thresholds,
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
        model_path = "/tmp/test_fisher_eval_bounded_model.pth"
        save_model(model, model_path)

        result = evaluate_fisher_thresholds(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=SPIKE_SHAPE,
            t_target=0.875,
            seed=42,
        )

        assert 0.0 <= result["baseline"]["accuracy"] <= 1.0
        assert 0.0 <= result["fisher"]["accuracy"] <= 1.0
