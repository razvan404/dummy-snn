import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from spiking.layers.integrate_and_fire import IntegrateAndFireLayer
from spiking.threshold import NormalInitialization


def make_layer(num_inputs=10, num_outputs=5, avg_threshold=5.0):
    """Create a minimal IntegrateAndFireLayer for testing."""
    threshold_init = NormalInitialization(
        avg_threshold=avg_threshold, min_threshold=1.0, std_dev=0.5
    )
    return IntegrateAndFireLayer(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        threshold_initialization=threshold_init,
        refractory_period=float("inf"),
    )


class FakeDataset:
    """Mimics SpikeEncodingDataset: each item is (times, label)."""

    def __init__(self, num_samples=5, shape=(2, 4, 4)):
        self.items = []
        for i in range(num_samples):
            times = torch.rand(shape)
            label = torch.tensor(i % 3)
            self.items.append((times, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def make_fake_dataloader(num_samples=5, shape=(2, 4, 4)):
    """Create a dataloader that yields (times, label) like SpikeEncodingDataset."""
    dataset = FakeDataset(num_samples=num_samples, shape=shape)
    return DataLoader(dataset, batch_size=None, shuffle=False)


class TestExtractFeatures:
    def test_returns_correct_shapes(self):
        from spiking.evaluation.feature_extraction import extract_features

        torch.manual_seed(42)
        shape = (2, 4, 4)
        num_inputs = 2 * 4 * 4  # 32
        num_outputs = 5
        num_samples = 5

        layer = make_layer(num_inputs=num_inputs, num_outputs=num_outputs)
        loader = make_fake_dataloader(num_samples=num_samples, shape=shape)

        X, y = extract_features(layer, loader, shape)

        assert X.shape == (num_samples, num_outputs)
        assert y.shape == (num_samples,)

    def test_values_in_valid_range(self):
        from spiking.evaluation.feature_extraction import extract_features

        torch.manual_seed(42)
        shape = (2, 4, 4)
        num_inputs = 2 * 4 * 4
        layer = make_layer(num_inputs=num_inputs, num_outputs=5)
        loader = make_fake_dataloader(num_samples=3, shape=shape)

        X, y = extract_features(layer, loader, shape)

        assert np.all(X >= 0.0), "Features should be >= 0"
        assert np.all(X <= 1.0), "Features should be <= 1"

    def test_model_is_reset_between_samples(self):
        from spiking.evaluation.feature_extraction import extract_features

        torch.manual_seed(42)
        shape = (2, 4, 4)
        num_inputs = 2 * 4 * 4
        layer = make_layer(num_inputs=num_inputs, num_outputs=5)
        loader = make_fake_dataloader(num_samples=3, shape=shape)

        extract_features(layer, loader, shape)

        # After processing, spike_times should be reset (all inf)
        assert torch.all(torch.isinf(layer.spike_times))

    def test_early_stopping_breaks_when_all_spiked(self):
        """Model with very low thresholds should trigger early stopping."""
        from spiking.evaluation.feature_extraction import extract_features

        torch.manual_seed(42)
        shape = (2, 4, 4)
        num_inputs = 2 * 4 * 4
        # Very low thresholds so all neurons spike early
        layer = make_layer(num_inputs=num_inputs, num_outputs=3, avg_threshold=0.01)
        loader = make_fake_dataloader(num_samples=2, shape=shape)

        # Count how many times forward is called
        original_forward = layer.forward
        call_count = [0]

        def counting_forward(*args, **kwargs):
            call_count[0] += 1
            return original_forward(*args, **kwargs)

        layer.forward = counting_forward
        X_early, _ = extract_features(layer, loader, shape)

        # Reset and run with high thresholds (no early stopping possible)
        layer_high = make_layer(
            num_inputs=num_inputs, num_outputs=3, avg_threshold=100.0
        )
        loader2 = make_fake_dataloader(num_samples=2, shape=shape)
        original_forward2 = layer_high.forward
        call_count_high = [0]

        def counting_forward2(*args, **kwargs):
            call_count_high[0] += 1
            return original_forward2(*args, **kwargs)

        layer_high.forward = counting_forward2
        extract_features(layer_high, loader2, shape)

        # Low-threshold model should have fewer forward calls due to early stopping
        assert call_count[0] < call_count_high[0], (
            f"Early stopping should reduce forward calls: "
            f"low_thresh={call_count[0]} vs high_thresh={call_count_high[0]}"
        )


class TestComputeMetrics:
    def test_returns_expected_keys(self):
        from spiking.evaluation.eval_utils import compute_metrics

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        result = compute_metrics(y_true, y_pred)

        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result

    def test_perfect_predictions(self):
        from spiking.evaluation.eval_utils import compute_metrics

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        result = compute_metrics(y_true, y_pred)

        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0

    def test_no_side_effects(self):
        """compute_metrics should not print or create plots."""
        from spiking.evaluation.eval_utils import compute_metrics
        import io
        import sys

        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 0])

        captured = io.StringIO()
        sys.stdout = captured
        try:
            compute_metrics(y_true, y_pred)
        finally:
            sys.stdout = sys.__stdout__

        assert captured.getvalue() == "", "compute_metrics should not print anything"


class TestEvaluateClassifier:
    def test_returns_train_and_val_metrics(self):
        from spiking.evaluation.eval_classifier import evaluate_classifier

        np.random.seed(42)
        X_train = np.random.rand(20, 10)
        y_train = np.array([0, 1] * 10)
        X_test = np.random.rand(10, 10)
        y_test = np.array([0, 1] * 5)

        train_metrics, val_metrics = evaluate_classifier(
            X_train, y_train, X_test, y_test
        )

        assert isinstance(train_metrics, dict)
        assert isinstance(val_metrics, dict)
        assert "accuracy" in train_metrics
        assert "accuracy" in val_metrics

    def test_custom_classifier(self):
        from spiking.evaluation.eval_classifier import evaluate_classifier
        from sklearn.svm import LinearSVC

        np.random.seed(42)
        X_train = np.random.rand(20, 10)
        y_train = np.array([0, 1] * 10)
        X_test = np.random.rand(10, 10)
        y_test = np.array([0, 1] * 5)

        classifier = LinearSVC(max_iter=20000)
        train_metrics, val_metrics = evaluate_classifier(
            X_train, y_train, X_test, y_test, classifier=classifier
        )

        assert isinstance(train_metrics, dict)
        assert isinstance(val_metrics, dict)


class TestIterateSpikes:
    def test_yields_one_entry_per_unique_time(self):
        """iterate_spikes should yield one entry per unique spike time."""
        from spiking import iterate_spikes

        times = torch.tensor([[[0.1, 0.2], [0.3, float("inf")]]])
        results = list(iterate_spikes(times))
        assert len(results) == 3

        for incoming_spikes, current_time, dt in results:
            assert incoming_spikes.shape == times.shape

    def test_empty_for_all_inf(self):
        """iterate_spikes should yield nothing when all times are inf."""
        from spiking import iterate_spikes

        times = torch.full((1, 2, 2), float("inf"))
        results = list(iterate_spikes(times))
        assert len(results) == 0

    def test_spike_frame_values_and_delta_times(self):
        """iterate_spikes should produce correct spike frames, times, and deltas."""
        from spiking import iterate_spikes

        # Two pixels spike at t=0.1, one at t=0.3, one is inf (no spike)
        times = torch.tensor([[[0.1, 0.3], [0.1, float("inf")]]])
        # Clone frames because iterate_spikes reuses an internal buffer.
        results = [(f.clone(), t, d) for f, t, d in iterate_spikes(times)]

        assert len(results) == 2

        # First frame: t=0.1, two spikes active
        frame0, time0, dt0 = results[0]
        assert time0 == pytest.approx(0.1, abs=1e-6)
        assert dt0 == pytest.approx(0.1, abs=1e-6)  # 0.1 - 0.0
        expected0 = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        assert torch.equal(frame0, expected0)

        # Second frame: t=0.3, one spike active
        frame1, time1, dt1 = results[1]
        assert time1 == pytest.approx(0.3, abs=1e-6)
        assert dt1 == pytest.approx(0.2, abs=1e-6)  # 0.3 - 0.1
        expected1 = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
        assert torch.equal(frame1, expected1)
