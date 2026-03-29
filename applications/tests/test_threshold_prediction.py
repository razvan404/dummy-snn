import numpy as np
import pytest
import torch

from spiking.layers import IntegrateAndFireLayer
from spiking.threshold import NormalInitialization


# ---- Fixtures ----


@pytest.fixture
def small_layer():
    """Create a small IF layer for testing."""
    layer = IntegrateAndFireLayer(
        num_inputs=32,
        num_outputs=8,
        threshold_initialization=NormalInitialization(
            avg_threshold=5.0, std_dev=1.0, min_threshold=1.0
        ),
        refractory_period=float("inf"),
    )
    return layer


@pytest.fixture
def random_input_times():
    """Random input spike times (B=50, I=32), discretized to 64 bins."""
    times = torch.rand(50, 32)
    # Discretize to 64 bins
    times = (times * 63).long().float() / 63.0
    # Some inputs don't spike
    mask = torch.rand(50, 32) > 0.8
    times[mask] = float("inf")
    return times


@pytest.fixture
def vmax_and_spikes(small_layer, random_input_times):
    """Compute V_max and spike times for the small layer."""
    spike_times, cum_potential = small_layer.infer_spike_times_and_potentials_batch(
        random_input_times
    )
    return cum_potential.numpy(), spike_times.numpy()


# ---- NeuronTracker Tests ----


class TestNeuronTracker:
    def test_shapes(self):
        from spiking.metrics import NeuronTracker

        tracker = NeuronTracker(n_neurons=16, n_epochs=10, n_classes=5)
        assert tracker.threshold_history.shape == (10, 16)
        assert tracker.win_counts.shape == (10, 16)
        assert tracker.win_counts_per_class.shape == (10, 16, 5)
        assert tracker.cumulative_stdp_mag.shape == (10, 16)

    def test_win_counting(self):
        from spiking.metrics import NeuronTracker

        tracker = NeuronTracker(n_neurons=4, n_epochs=2, n_classes=3)
        tracker._current_epoch = 0

        class FakeLearner:
            neurons_to_learn = torch.tensor([1])

        learner = FakeLearner()
        tracker.on_batch(learner, label=0, dw=0.5)
        tracker.on_batch(learner, label=1, dw=0.3)

        learner.neurons_to_learn = torch.tensor([2])
        tracker.on_batch(learner, label=2, dw=0.1)

        assert tracker.win_counts[0, 1] == 2
        assert tracker.win_counts[0, 2] == 1
        assert tracker.win_counts_per_class[0, 1, 0] == 1
        assert tracker.win_counts_per_class[0, 1, 1] == 1
        assert tracker.win_counts_per_class[0, 2, 2] == 1
        assert tracker.cumulative_stdp_mag[0, 1] == pytest.approx(0.8, abs=1e-6)

    def test_trajectory_features_shapes(self):
        from spiking.metrics import NeuronTracker

        tracker = NeuronTracker(n_neurons=8, n_epochs=20, n_classes=10)
        # Simulate some data
        tracker.threshold_history = np.random.rand(20, 8) * 10
        tracker.win_counts_per_class = np.random.randint(0, 5, (20, 8, 10)).astype(
            float
        )
        tracker.cumulative_stdp_mag = np.random.rand(20, 8)
        # Add weight snapshots
        for e in [9, 19]:
            tracker.weight_snapshots.append(np.random.rand(8, 32))
            tracker.weight_snapshot_epochs.append(e)

        features = tracker.compute_trajectory_features()
        for key, val in features.items():
            assert val.shape == (8,), f"Feature {key} has wrong shape: {val.shape}"

    def test_no_winners_handled(self):
        from spiking.metrics import NeuronTracker

        tracker = NeuronTracker(n_neurons=4, n_epochs=2, n_classes=3)
        tracker._current_epoch = 0

        class FakeLearner:
            neurons_to_learn = torch.tensor([], dtype=torch.long)

        learner = FakeLearner()
        tracker.on_batch(learner, label=0, dw=0.0)
        assert tracker.win_counts[0].sum() == 0


# ---- V_max / Distribution Features Tests ----


class TestDistributionFeatures:
    def test_compute_vmax_batch_shape(self, small_layer, random_input_times):
        from spiking.metrics import compute_vmax_batch

        vmax, spike_times = compute_vmax_batch(small_layer, random_input_times)
        assert vmax.shape == (50, 8)
        assert spike_times.shape == (50, 8)

    def test_vmax_consistency_with_layer(self, small_layer, random_input_times):
        """V_max from compute_vmax_batch should match layer's potentials_batch."""
        from spiking.metrics import compute_vmax_batch

        vmax, spike_times = compute_vmax_batch(small_layer, random_input_times)
        spike_ref, pot_ref = small_layer.infer_spike_times_and_potentials_batch(
            random_input_times
        )
        np.testing.assert_allclose(
            vmax.numpy(),
            pot_ref.numpy(),
            atol=1e-5,
            err_msg="V_max should match layer's cumulative potentials",
        )
        np.testing.assert_allclose(
            spike_times.numpy(),
            spike_ref.numpy(),
            atol=1e-5,
            err_msg="Spike times should match",
        )

    def test_distribution_features_shapes(self, vmax_and_spikes, small_layer):
        from spiking.metrics import compute_distribution_features

        vmax, _ = vmax_and_spikes
        thresholds = small_layer.thresholds.detach().numpy()
        features = compute_distribution_features(vmax, thresholds)

        expected_keys = [
            "v_mean",
            "v_std",
            "v_median",
            "v_skewness",
            "v_kurtosis",
            "v_quantile_99",
            "v_quantile_995",
            "v_quantile_998",
            "v_bimodality",
            "v_firing_ratio",
            "v_entropy_binned",
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"
            assert features[key].shape == (8,), f"Feature {key} wrong shape"

    def test_firing_ratio_range(self, vmax_and_spikes, small_layer):
        from spiking.metrics import compute_distribution_features

        vmax, _ = vmax_and_spikes
        thresholds = small_layer.thresholds.detach().numpy()
        features = compute_distribution_features(vmax, thresholds)
        assert np.all(features["v_firing_ratio"] >= 0.0)
        assert np.all(features["v_firing_ratio"] <= 1.0)


# ---- Inter-neuron Features Tests ----


class TestInterNeuronFeatures:
    def test_shapes(self, vmax_and_spikes, small_layer):
        from spiking.metrics import compute_inter_neuron_features

        vmax, spikes = vmax_and_spikes
        weights = small_layer.weights.detach().numpy()
        thresholds = small_layer.thresholds.detach().numpy()
        features = compute_inter_neuron_features(weights, vmax, spikes, thresholds)

        expected_keys = [
            "weight_overlap_mean",
            "weight_overlap_max",
            "weight_overlap_top5",
            "competition_margin_mean",
            "competition_margin_std",
            "narrow_win_fraction",
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"
            assert features[key].shape == (8,), f"Feature {key} wrong shape"

    def test_weight_overlap_range(self, small_layer):
        from spiking.metrics import compute_inter_neuron_features

        weights = small_layer.weights.detach().numpy()
        # Use dummy V_max and spike times
        vmax = np.random.rand(20, 8)
        spikes = np.full((20, 8), float("inf"))
        thresholds = np.ones(8)
        features = compute_inter_neuron_features(weights, vmax, spikes, thresholds)

        # Cosine similarity should be in [-1, 1]
        assert np.all(features["weight_overlap_mean"] >= -1.0)
        assert np.all(features["weight_overlap_mean"] <= 1.0)
        assert np.all(features["weight_overlap_max"] >= -1.0)
        assert np.all(features["weight_overlap_max"] <= 1.0)


# ---- Cached Coordinate Descent Tests ----


class TestCachedOptimizer:
    def test_spike_time_resolution_matches_layer(self, small_layer, random_input_times):
        """Verify cached spike times match the layer's analytical computation."""
        from applications.threshold_research.neuron_perturbation import (
            precompute_cumulative_potentials,
            spike_times_from_potentials,
        )

        weights = small_layer.weights.detach()
        thresholds = small_layer.thresholds.detach()

        cum_pot, boundary = precompute_cumulative_potentials(
            random_input_times, weights
        )

        # Resolve spike times from cached potentials
        num_neurons = weights.shape[0]
        B = random_input_times.shape[0]
        cached_spikes = torch.full((B, num_neurons), float("inf"))
        for i in range(num_neurons):
            cached_spikes[:, i] = spike_times_from_potentials(
                cum_pot[:, i, :], boundary, thresholds[i].item()
            )

        # Compare with layer's direct computation
        direct_spikes = small_layer.infer_spike_times_batch(random_input_times)

        np.testing.assert_allclose(
            cached_spikes.numpy(),
            direct_spikes.numpy(),
            atol=1e-5,
            err_msg="Cached spike times should match layer's analytical computation",
        )

    def test_evaluate_thresholds_deterministic(self, small_layer, random_input_times):
        """Same thresholds should give same accuracy."""
        from applications.threshold_research.neuron_perturbation import (
            collect_input_times,
            compute_features_with_thresholds,
            precompute_cumulative_potentials,
        )
        from sklearn.linear_model import RidgeClassifier
        from spiking.evaluation.eval_utils import compute_metrics

        weights = small_layer.weights.detach()
        thresholds = small_layer.thresholds.detach()

        cum_pot, boundary = precompute_cumulative_potentials(
            random_input_times, weights
        )
        labels = np.random.randint(0, 5, size=random_input_times.shape[0])

        X = compute_features_with_thresholds(cum_pot, boundary, thresholds, None)
        clf = RidgeClassifier()
        clf.fit(X, labels)
        pred = clf.predict(X)
        acc1 = compute_metrics(labels, pred)["accuracy"]

        # Second evaluation with same thresholds
        X2 = compute_features_with_thresholds(cum_pot, boundary, thresholds, None)
        pred2 = clf.predict(X2)
        acc2 = compute_metrics(labels, pred2)["accuracy"]

        assert acc1 == acc2


# ---- Analytical Baselines Tests ----


class TestAnalyticalBaselines:
    def test_quantile_monotonic(self):
        from applications.threshold_prediction.analytical_baselines import (
            quantile_thresholds,
        )

        vmax = np.random.rand(100, 8)
        th_low = quantile_thresholds(vmax, 0.01)  # higher p → lower threshold
        th_high = quantile_thresholds(vmax, 0.1)
        # Higher firing probability → lower thresholds
        assert np.all(th_high <= th_low + 1e-10)

    def test_otsu_within_range(self):
        from applications.threshold_prediction.analytical_baselines import (
            otsu_per_neuron,
        )

        vmax = np.random.rand(100, 4) * 10 + 2
        thresholds = otsu_per_neuron(vmax)
        for i in range(4):
            assert thresholds[i] >= vmax[:, i].min()
            assert thresholds[i] <= vmax[:, i].max()

    def test_uniform_mean_is_mean(self):
        from applications.threshold_prediction.analytical_baselines import uniform_mean

        trained = np.array([1.0, 2.0, 3.0, 4.0])
        result = uniform_mean(trained)
        np.testing.assert_allclose(result, 2.5)

    def test_mean_plus_k_sigma_ordering(self):
        from applications.threshold_prediction.analytical_baselines import (
            mean_plus_k_sigma,
        )

        vmax = np.random.rand(100, 4)
        th1 = mean_plus_k_sigma(vmax, 1.0)
        th2 = mean_plus_k_sigma(vmax, 2.0)
        assert np.all(th2 >= th1 - 1e-10)


# ---- Threshold Predictor Tests ----


class TestThresholdPredictor:
    def test_build_feature_matrix(self):
        from applications.threshold_prediction.threshold_predictor import (
            build_feature_matrix,
        )

        traj = {"a": np.ones(8), "b": np.zeros(8)}
        dist = {"c": np.random.rand(8)}
        X, names = build_feature_matrix(
            trajectory_features=traj, distribution_features=dist
        )
        assert X.shape[0] == 8
        assert X.shape[1] >= 3  # at least a, b, c
        assert "trajectory.a" in names
        assert "trajectory.b" in names
        assert "distribution.c" in names

    def test_fit_and_evaluate_runs(self):
        from applications.threshold_prediction.threshold_predictor import (
            fit_and_evaluate,
        )

        np.random.seed(42)
        X = np.random.rand(30, 5)
        y = np.random.rand(30)
        groups = np.repeat([1, 2, 3], 10)

        results = fit_and_evaluate(X, y, groups, [f"f{i}" for i in range(5)])
        assert "per_model" in results
        assert "best_model_name" in results
        assert "feature_importance" in results
        for model_name in ["Ridge", "GBR", "RF", "MLP"]:
            assert model_name in results["per_model"]

    def test_feature_group_ablation_runs(self):
        from applications.threshold_prediction.threshold_predictor import (
            feature_group_ablation,
        )

        np.random.seed(42)
        X = np.random.rand(30, 5)
        y = np.random.rand(30)
        groups = np.repeat([1, 2, 3], 10)
        names = ["group_a.f1", "group_a.f2", "group_b.f3", "group_b.f4", "group_b.f5"]

        results = feature_group_ablation(X, y, groups, names)
        group_names = [r["group"] for r in results]
        assert "group_a" in group_names
        assert "group_b" in group_names
        assert "all" in group_names
