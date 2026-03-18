import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer
from spiking.learning.conv_learner import ConvLearner
from spiking.learning.multiplicative_stdp import MultiplicativeSTDP
from spiking.learning.wta import WinnerTakesAll
from spiking.threshold import ConstantInitialization
from spiking.training.conv_trainer import ConvUnsupervisedTrainer


def make_layer(threshold=1.0):
    init = ConstantInitialization(threshold)
    return ConvIntegrateAndFireLayer(
        in_channels=2,
        num_filters=4,
        kernel_size=3,
        padding=0,
        threshold_initialization=init,
        refractory_period=float("inf"),
    )


def make_learner(layer):
    stdp = MultiplicativeSTDP(learning_rate=0.1, beta=1.0)
    wta = WinnerTakesAll()
    return ConvLearner(layer, stdp, competition=wta)


def make_loader(n=10, C=2, H=8, W=8):
    """Create a fake DataLoader with spike time tensors."""
    torch.manual_seed(42)
    times = torch.rand(n, C, H, W)
    times[times > 0.7] = float("inf")
    labels = torch.randint(0, 10, (n,))
    dataset = TensorDataset(times, labels)
    return DataLoader(dataset, batch_size=None, shuffle=False)


class TestConvTrainerConstruction:
    def test_create(self):
        layer = make_layer()
        learner = make_learner(layer)
        trainer = ConvUnsupervisedTrainer(layer, learner, image_shape=(2, 8, 8))
        assert trainer.model is layer
        assert trainer.learner is learner


class TestConvTrainerStepBatch:
    def test_step_batch_returns_dw(self):
        layer = make_layer(threshold=0.5)
        layer.weights.data.fill_(0.5)
        learner = make_learner(layer)
        trainer = ConvUnsupervisedTrainer(layer, learner, image_shape=(2, 8, 8))
        layer.train()

        times = torch.rand(2, 8, 8)
        times[times > 0.7] = float("inf")
        dw = trainer.step_batch(0, times)
        assert isinstance(dw, float)

    def test_step_batch_preserves_spatial_shape(self):
        """Input should not be flattened — spatial structure preserved."""
        layer = make_layer(threshold=0.5)
        layer.weights.data.fill_(0.5)
        learner = make_learner(layer)
        trainer = ConvUnsupervisedTrainer(layer, learner, image_shape=(2, 8, 8))
        layer.train()

        times = torch.rand(2, 8, 8)
        times[times > 0.7] = float("inf")
        trainer.step_batch(0, times)
        # After forward pass, spike times should be 3D
        assert layer.spike_times.dim() == 3

    def test_step_batch_resets_after(self):
        layer = make_layer(threshold=0.5)
        layer.weights.data.fill_(0.5)
        learner = make_learner(layer)
        trainer = ConvUnsupervisedTrainer(layer, learner, image_shape=(2, 8, 8))
        layer.train()

        times = torch.rand(2, 8, 8)
        times[times > 0.7] = float("inf")
        trainer.step_batch(0, times)
        # After step_batch, model should be reset
        assert torch.isinf(layer.spike_times).all()


class TestConvTrainerStepLoader:
    def test_step_loader_processes_all_samples(self):
        layer = make_layer(threshold=0.5)
        layer.weights.data.fill_(0.5)
        learner = make_learner(layer)
        trainer = ConvUnsupervisedTrainer(layer, learner, image_shape=(2, 8, 8))
        loader = make_loader(n=5)

        batch_count = []
        trainer.on_batch_end = lambda idx, dw, split: batch_count.append(idx)
        trainer.step_loader(loader, split="train")
        assert len(batch_count) == 5

    def test_step_loader_eval_mode(self):
        layer = make_layer(threshold=0.5)
        layer.weights.data.fill_(0.5)
        learner = make_learner(layer)
        trainer = ConvUnsupervisedTrainer(layer, learner, image_shape=(2, 8, 8))
        loader = make_loader(n=3)

        weights_before = layer.weights.data.clone()
        trainer.step_loader(loader, split="val")
        # In eval mode, weights should not change
        assert torch.allclose(layer.weights.data, weights_before)


class TestConvTrainerStepEpoch:
    def test_step_epoch_decays_learning_rate(self):
        layer = make_layer()
        stdp = MultiplicativeSTDP(learning_rate=0.1, decay_factor=0.5)
        learner = ConvLearner(layer, stdp)
        trainer = ConvUnsupervisedTrainer(layer, learner, image_shape=(2, 8, 8))
        trainer.step_epoch()
        assert stdp.learning_rate == pytest.approx(0.05)


class TestConvTrainerOnBatchEnd:
    def test_callback_called(self):
        layer = make_layer(threshold=0.5)
        layer.weights.data.fill_(0.5)
        learner = make_learner(layer)
        calls = []
        trainer = ConvUnsupervisedTrainer(
            layer,
            learner,
            image_shape=(2, 8, 8),
            on_batch_end=lambda idx, dw, split: calls.append((idx, split)),
        )
        layer.train()

        times = torch.rand(2, 8, 8)
        times[times > 0.7] = float("inf")
        trainer.step_batch(0, times, split="train")
        assert len(calls) == 1
        assert calls[0] == (0, "train")
