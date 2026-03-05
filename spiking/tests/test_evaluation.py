import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from spiking.layers.integrate_and_fire import IntegrateAndFireLayer
from spiking.threshold import ConstantInitialization, NormalInitialization


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

    def test_matches_iterative_forward_pass(self):
        """extract_features results must match the iterative forward-pass loop."""
        from spiking import iterate_spikes
        from spiking.evaluation.feature_extraction import extract_features

        torch.manual_seed(42)
        shape = (2, 4, 4)
        num_inputs = 2 * 4 * 4
        layer = make_layer(num_inputs=num_inputs, num_outputs=5)
        dataset = FakeDataset(num_samples=5, shape=shape)

        # Compute expected features using iterative forward pass
        expected_X = []
        layer.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                times, label = dataset[i]
                flat_times = times.flatten()
                for incoming_spikes, current_time, dt in iterate_spikes(flat_times):
                    layer.forward(incoming_spikes, current_time=current_time, dt=dt)
                    if torch.all(torch.isfinite(layer.spike_times)):
                        break
                expected_X.append(
                    torch.clamp(1.0 - layer.spike_times, min=0, max=1.0).numpy()
                )
                layer.reset()
        expected_X = np.array(expected_X)

        loader = DataLoader(dataset, batch_size=None, shuffle=False)
        actual_X, _ = extract_features(layer, loader, shape)

        np.testing.assert_allclose(actual_X, expected_X, atol=1e-6)

    def test_without_t_target_uses_linear_inversion(self):
        """Without t_target, features = clamp(1 - spike_times, 0, 1)."""
        from spiking.evaluation.feature_extraction import extract_features

        torch.manual_seed(42)
        shape = (2, 4, 4)
        num_inputs = 2 * 4 * 4
        layer = make_layer(num_inputs=num_inputs, num_outputs=5)
        loader = make_fake_dataloader(num_samples=3, shape=shape)

        X_default, _ = extract_features(layer, loader, shape)
        X_none, _ = extract_features(layer, loader, shape, t_target=None)

        np.testing.assert_array_equal(X_default, X_none)

    def test_with_t_target_uses_eq10(self):
        """With t_target, features = clamp(1 - (t - t_target) / (1 - t_target), 0, 1)."""
        from spiking.evaluation.feature_extraction import extract_features

        torch.manual_seed(42)
        shape = (2, 4, 4)
        num_inputs = 2 * 4 * 4
        # Low threshold so neurons fire early (before t_target=0.7)
        layer = make_layer(num_inputs=num_inputs, num_outputs=5, avg_threshold=0.01)
        loader = make_fake_dataloader(num_samples=3, shape=shape)

        X_eq10, _ = extract_features(layer, loader, shape, t_target=0.7)
        X_linear, _ = extract_features(layer, loader, shape, t_target=None)

        # With low thresholds, neurons fire early → spike_times << 0.7
        # Eq 10 clamps everything before t_target to 1.0
        # Linear formula gives 1 - spike_time < 1.0
        # So Eq 10 values should be >= linear values
        assert np.all(X_eq10 >= X_linear - 1e-6)

    def test_eq10_specific_values(self):
        """Verify Eq 10 formula against hand-computed values."""
        from spiking.evaluation.feature_extraction import spike_times_to_features

        t_target = 0.7
        spike_times = torch.tensor([0.5, 0.7, 0.8, 1.0, float("inf")])

        features = spike_times_to_features(spike_times, t_target=t_target)

        # t=0.5 < t_target → clamped to 1.0
        assert features[0].item() == pytest.approx(1.0)
        # t=0.7 = t_target → 1 - 0/0.3 = 1.0
        assert features[1].item() == pytest.approx(1.0)
        # t=0.8 → 1 - 0.1/0.3 = 0.6667
        assert features[2].item() == pytest.approx(2.0 / 3.0, abs=1e-4)
        # t=1.0 → 1 - 0.3/0.3 = 0.0
        assert features[3].item() == pytest.approx(0.0)
        # t=inf → 0.0
        assert features[4].item() == pytest.approx(0.0)

    def test_linear_specific_values(self):
        """Verify linear formula against hand-computed values."""
        from spiking.evaluation.feature_extraction import spike_times_to_features

        spike_times = torch.tensor([0.0, 0.3, 0.7, 1.0, float("inf")])

        features = spike_times_to_features(spike_times, t_target=None)

        assert features[0].item() == pytest.approx(1.0)
        assert features[1].item() == pytest.approx(0.7)
        assert features[2].item() == pytest.approx(0.3)
        assert features[3].item() == pytest.approx(0.0)
        assert features[4].item() == pytest.approx(0.0)

    def test_uses_analytical_path_without_forward_calls(self):
        """extract_features should use infer_spike_times, not iterative forward."""
        from spiking.evaluation.feature_extraction import extract_features

        torch.manual_seed(42)
        shape = (2, 4, 4)
        num_inputs = 2 * 4 * 4
        layer = make_layer(num_inputs=num_inputs, num_outputs=3, avg_threshold=0.01)
        loader = make_fake_dataloader(num_samples=2, shape=shape)

        original_forward = layer.forward
        call_count = [0]

        def counting_forward(*args, **kwargs):
            call_count[0] += 1
            return original_forward(*args, **kwargs)

        layer.forward = counting_forward
        extract_features(layer, loader, shape)

        assert call_count[0] == 0, (
            f"extract_features should use analytical path, "
            f"but forward was called {call_count[0]} times"
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


class TestInferSpikeTimes:
    def _make_layer_with_params(self, weights, thresholds, refractory_period=float("inf")):
        """Create a layer with exact weights and thresholds."""
        layer = IntegrateAndFireLayer(
            num_inputs=weights.shape[1],
            num_outputs=weights.shape[0],
            threshold_initialization=ConstantInitialization(1.0),
            refractory_period=refractory_period,
        )
        with torch.no_grad():
            layer.weights.copy_(weights)
            layer.thresholds.copy_(thresholds)
        return layer

    def test_single_neuron_crosses_threshold(self):
        weights = torch.tensor([[2.0]])
        thresholds = torch.tensor([1.0])
        layer = self._make_layer_with_params(weights, thresholds)

        result = layer.infer_spike_times(torch.tensor([0.5]))

        assert result.shape == (1,)
        assert result[0].item() == pytest.approx(0.5)

    def test_below_threshold_returns_inf(self):
        weights = torch.tensor([[0.5]])
        thresholds = torch.tensor([1.0])
        layer = self._make_layer_with_params(weights, thresholds)

        result = layer.infer_spike_times(torch.tensor([0.3]))

        assert torch.isinf(result[0])

    def test_cumulative_crossing(self):
        weights = torch.tensor([[0.6, 0.6]])
        thresholds = torch.tensor([1.0])
        layer = self._make_layer_with_params(weights, thresholds)

        result = layer.infer_spike_times(torch.tensor([0.1, 0.3]))

        # After t=0.1: V = 0.6 < 1.0
        # After t=0.3: V = 1.2 >= 1.0 → spike at 0.3
        assert result[0].item() == pytest.approx(0.3)

    def test_simultaneous_inputs_grouped(self):
        weights = torch.tensor([[0.6, 0.6]])
        thresholds = torch.tensor([1.0])
        layer = self._make_layer_with_params(weights, thresholds)

        result = layer.infer_spike_times(torch.tensor([0.2, 0.2]))

        # Both arrive at t=0.2: V = 1.2 >= 1.0 → spike at 0.2
        assert result[0].item() == pytest.approx(0.2)

    def test_all_inf_inputs_returns_all_inf(self):
        weights = torch.tensor([[1.0, 1.0]])
        thresholds = torch.tensor([0.5])
        layer = self._make_layer_with_params(weights, thresholds)

        result = layer.infer_spike_times(torch.tensor([float("inf"), float("inf")]))

        assert torch.all(torch.isinf(result))

    def test_equivalence_with_iterative_forward(self):
        """Analytical spike times must match iterative forward pass."""
        from spiking import iterate_spikes

        for seed in range(5):
            torch.manual_seed(seed)
            layer = make_layer(num_inputs=32, num_outputs=10, avg_threshold=5.0)
            input_times = torch.rand(32)

            # Iterative forward pass
            layer.reset()
            for incoming_spikes, current_time, dt in iterate_spikes(input_times):
                layer.forward(incoming_spikes, current_time=current_time, dt=dt)
            iterative_times = layer.spike_times.clone()
            layer.reset()

            # Analytical
            analytical_times = layer.infer_spike_times(input_times)

            assert torch.allclose(iterative_times, analytical_times, atol=1e-6), (
                f"Seed {seed}: iterative={iterative_times} vs analytical={analytical_times}"
            )

    def test_sequential_equivalence(self):
        """Multi-layer analytical must match iterative."""
        from spiking import iterate_spikes
        from spiking.layers.sequential import SpikingSequential

        for seed in range(5):
            torch.manual_seed(seed)
            layer1 = make_layer(num_inputs=32, num_outputs=16, avg_threshold=5.0)
            layer2 = make_layer(num_inputs=16, num_outputs=10, avg_threshold=5.0)
            model = SpikingSequential(layer1, layer2)

            input_times = torch.rand(32)

            # Iterative
            model.reset()
            for incoming_spikes, current_time, dt in iterate_spikes(input_times):
                model.forward(incoming_spikes, current_time=current_time, dt=dt)
                if torch.all(torch.isfinite(model.spike_times)):
                    break
            iterative_times = model.spike_times.clone()
            model.reset()

            # Analytical
            analytical_times = model.infer_spike_times(input_times)

            assert torch.allclose(iterative_times, analytical_times, atol=1e-6), (
                f"Seed {seed}: iterative={iterative_times} vs analytical={analytical_times}"
            )

class TestInferSpikeTimesBatch:
    def _make_layer_with_params(self, weights, thresholds, refractory_period=float("inf")):
        """Create a layer with exact weights and thresholds."""
        layer = IntegrateAndFireLayer(
            num_inputs=weights.shape[1],
            num_outputs=weights.shape[0],
            threshold_initialization=ConstantInitialization(1.0),
            refractory_period=refractory_period,
        )
        with torch.no_grad():
            layer.weights.copy_(weights)
            layer.thresholds.copy_(thresholds)
        return layer

    def test_single_neuron_crosses_threshold(self):
        weights = torch.tensor([[2.0]])
        thresholds = torch.tensor([1.0])
        layer = self._make_layer_with_params(weights, thresholds)

        single = layer.infer_spike_times(torch.tensor([0.5]))
        batch = layer.infer_spike_times_batch(torch.tensor([[0.5]]))

        assert batch.shape == (1, 1)
        torch.testing.assert_close(batch[0], single)

    def test_below_threshold_returns_inf(self):
        weights = torch.tensor([[0.5]])
        thresholds = torch.tensor([1.0])
        layer = self._make_layer_with_params(weights, thresholds)

        single = layer.infer_spike_times(torch.tensor([0.3]))
        batch = layer.infer_spike_times_batch(torch.tensor([[0.3]]))

        assert torch.isinf(batch[0, 0])
        torch.testing.assert_close(batch[0], single)

    def test_cumulative_crossing(self):
        weights = torch.tensor([[0.6, 0.6]])
        thresholds = torch.tensor([1.0])
        layer = self._make_layer_with_params(weights, thresholds)

        single = layer.infer_spike_times(torch.tensor([0.1, 0.3]))
        batch = layer.infer_spike_times_batch(torch.tensor([[0.1, 0.3]]))

        torch.testing.assert_close(batch[0], single)

    def test_simultaneous_inputs_grouped(self):
        weights = torch.tensor([[0.6, 0.6]])
        thresholds = torch.tensor([1.0])
        layer = self._make_layer_with_params(weights, thresholds)

        single = layer.infer_spike_times(torch.tensor([0.2, 0.2]))
        batch = layer.infer_spike_times_batch(torch.tensor([[0.2, 0.2]]))

        torch.testing.assert_close(batch[0], single)

    def test_all_inf_inputs_returns_all_inf(self):
        weights = torch.tensor([[1.0, 1.0]])
        thresholds = torch.tensor([0.5])
        layer = self._make_layer_with_params(weights, thresholds)

        inp = torch.tensor([[float("inf"), float("inf")]])
        batch = layer.infer_spike_times_batch(inp)

        assert torch.all(torch.isinf(batch))

    def test_batch_matches_individual_across_seeds(self):
        """Process multiple random samples as a batch, verify each row matches single-sample."""
        for seed in range(5):
            torch.manual_seed(seed)
            layer = make_layer(num_inputs=32, num_outputs=10, avg_threshold=5.0)
            inputs = torch.rand(5, 32)

            batch_result = layer.infer_spike_times_batch(inputs)

            for i in range(5):
                single = layer.infer_spike_times(inputs[i])
                torch.testing.assert_close(batch_result[i], single, atol=1e-6, rtol=0)

    def test_sequential_batch_matches_individual(self):
        """Multi-layer batched must match per-sample analytical."""
        from spiking.layers.sequential import SpikingSequential

        for seed in range(5):
            torch.manual_seed(seed)
            layer1 = make_layer(num_inputs=32, num_outputs=16, avg_threshold=5.0)
            layer2 = make_layer(num_inputs=16, num_outputs=10, avg_threshold=5.0)
            model = SpikingSequential(layer1, layer2)

            inputs = torch.rand(5, 32)
            batch_result = model.infer_spike_times_batch(inputs)

            for i in range(5):
                single = model.infer_spike_times(inputs[i])
                torch.testing.assert_close(batch_result[i], single, atol=1e-6, rtol=0)

    def test_extract_features_batched_matches_original(self):
        """Batched extract_features must match per-sample infer_spike_times loop."""
        from spiking.evaluation.feature_extraction import extract_features

        torch.manual_seed(42)
        shape = (2, 4, 4)
        num_inputs = 2 * 4 * 4
        num_samples = 10
        layer = make_layer(num_inputs=num_inputs, num_outputs=5, avg_threshold=5.0)
        dataset = FakeDataset(num_samples=num_samples, shape=shape)

        # Ground truth: per-sample analytical inference
        expected_X = []
        for i in range(num_samples):
            times, label = dataset[i]
            spike_times = layer.infer_spike_times(times.flatten())
            expected_X.append(torch.clamp(1.0 - spike_times, min=0, max=1.0).numpy())
        expected_X = np.array(expected_X)

        loader = DataLoader(dataset, batch_size=None, shuffle=False)
        actual_X, _ = extract_features(layer, loader, shape)

        np.testing.assert_allclose(actual_X, expected_X, atol=1e-6)


class TestConvertToSpikes:
    def test_basic_conversion(self):
        from spiking.spike_convertor import convert_to_spikes

        times = torch.tensor([
            [[0.1, 0.3], [float("inf"), 0.2]],  # k=0
        ])
        spikes = convert_to_spikes(times)

        assert len(spikes) == 3
        # Should be sorted by time
        assert spikes[0].time == pytest.approx(0.1)
        assert spikes[1].time == pytest.approx(0.2)
        assert spikes[2].time == pytest.approx(0.3)
        # Check coordinates: Spike(x=j, y=i, z=k, time=...)
        assert (spikes[0].z, spikes[0].y, spikes[0].x) == (0, 0, 0)
        assert (spikes[1].z, spikes[1].y, spikes[1].x) == (0, 1, 1)
        assert (spikes[2].z, spikes[2].y, spikes[2].x) == (0, 0, 1)

    def test_all_inf_returns_empty(self):
        from spiking.spike_convertor import convert_to_spikes

        times = torch.full((2, 3, 3), float("inf"))
        assert convert_to_spikes(times) == []

    def test_multi_channel(self):
        from spiking.spike_convertor import convert_to_spikes

        times = torch.full((2, 2, 2), float("inf"))
        times[0, 0, 1] = 0.5
        times[1, 1, 0] = 0.3

        spikes = convert_to_spikes(times)
        assert len(spikes) == 2
        assert spikes[0].time == pytest.approx(0.3)
        assert spikes[0].z == 1
        assert spikes[1].time == pytest.approx(0.5)
        assert spikes[1].z == 0


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
