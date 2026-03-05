import torch
import pytest

from spiking.training.monitor import TrainingMonitor


class FakeModel:
    """Minimal stand-in for SpikingModule with the fields TrainingMonitor uses."""

    def __init__(self, num_outputs=100, thresholds=None, spike_times=None):
        self.num_outputs = num_outputs
        self.thresholds = (
            thresholds if thresholds is not None else torch.ones(num_outputs)
        )
        self.spike_times = (
            spike_times
            if spike_times is not None
            else torch.full((num_outputs,), float("inf"))
        )


class TestLogInterval:
    def test_default_log_interval_is_one(self):
        model = FakeModel()
        monitor = TrainingMonitor(model)
        assert monitor.log_interval == 1

    def test_log_interval_records_thresholds_every_batch_when_one(self):
        model = FakeModel(num_outputs=10)
        monitor = TrainingMonitor(model, log_interval=1)
        for i in range(5):
            monitor.log(split="train", dw=0.1)
        assert len(monitor.thresholds["mean"]) == 5
        assert len(monitor.thresholds["list"]) == 5

    def test_log_interval_skips_threshold_logging(self):
        model = FakeModel(num_outputs=10)
        monitor = TrainingMonitor(model, log_interval=3)
        for i in range(9):
            monitor.log(split="train", dw=0.1)
        # Batches 0,3,6 should log thresholds (3 out of 9)
        assert len(monitor.thresholds["mean"]) == 3
        assert len(monitor.thresholds["list"]) == 3

    def test_weight_diffs_always_recorded(self):
        model = FakeModel(num_outputs=10)
        monitor = TrainingMonitor(model, log_interval=5)
        for i in range(10):
            monitor.log(split="train", dw=float(i))
        assert len(monitor.weight_diffs["train"]) == 10

    def test_neuron_activity_always_recorded(self):
        model = FakeModel(num_outputs=5)
        # Make neuron 0 spike at time 0.5
        model.spike_times = torch.tensor(
            [0.5, float("inf"), float("inf"), float("inf"), float("inf")]
        )
        monitor = TrainingMonitor(model, log_interval=100)
        for i in range(10):
            monitor.log(split="train", dw=0.1)
        # Neuron 0 should have activity = 10
        assert monitor.neurons_activity[0].item() == 10.0

    def test_most_active_neurons_uses_activity_not_thresholds(self):
        """most_active_neurons() should return neurons with highest activity counts."""
        model = FakeModel(num_outputs=5)
        # Set thresholds so neuron 4 has the highest threshold
        model.thresholds = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])
        # Make neuron 0 spike repeatedly (highest activity)
        model.spike_times = torch.tensor(
            [0.1, float("inf"), float("inf"), float("inf"), float("inf")]
        )
        monitor = TrainingMonitor(model)

        # Log several batches so neuron 0 accumulates activity
        for _ in range(10):
            monitor.log(split="train", dw=0.1)

        result = monitor.most_active_neurons(num_neurons=1)
        # Should return neuron 0 (most active), NOT neuron 4 (highest threshold)
        assert result[0].item() == 0

    def test_non_train_split_unaffected_by_log_interval(self):
        model = FakeModel(num_outputs=10)
        monitor = TrainingMonitor(model, log_interval=1)
        for i in range(5):
            monitor.log(split="val", dw=0.1)
        assert len(monitor.weight_diffs["val"]) == 5
        # Thresholds not logged for non-train splits regardless
        assert len(monitor.thresholds["mean"]) == 0
