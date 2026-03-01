import torch
import pytest
from torch.utils.data import DataLoader

from applications.datasets import MnistSubsetDataset


IMAGE_SHAPE = (8, 8)
SPIKE_SHAPE = (2, *IMAGE_SHAPE)
NUM_INPUTS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * 2  # 128

SETUP = {
    "threshold_init": {
        "avg_threshold": 5.0,
        "min_threshold": 1.0,
        "std_dev": 0.5,
    },
    "threshold_adaptation": {
        "min_threshold": 1.0,
        "learning_rate": 5.0,
        "decay_factor": 1.0,
    },
    "stdp": {
        "tau_pre": 0.1,
        "tau_post": 0.1,
        "max_pre_spike_time": 1.0,
        "learning_rate": 0.1,
        "decay_factor": 1.0,
    },
}

TINY_ARCHITECTURE = [16, 8, 4]


@pytest.fixture(scope="module")
def train_loader():
    dataset = MnistSubsetDataset("data/mnist-subset", "train", image_shape=IMAGE_SHAPE)
    return DataLoader(dataset, batch_size=None, shuffle=True)


@pytest.fixture(scope="module")
def val_loader():
    dataset = MnistSubsetDataset("data/mnist-subset", "test", image_shape=IMAGE_SHAPE)
    return DataLoader(dataset, batch_size=None, shuffle=False)


class TestCreateModel:
    def test_returns_correct_number_of_layers(self):
        from applications.deep_linear.model import create_model

        model = create_model(SETUP, NUM_INPUTS, TINY_ARCHITECTURE)
        assert len(model.layers) == 3

    def test_layer_sizes_match_architecture(self):
        from applications.deep_linear.model import create_model

        model = create_model(SETUP, NUM_INPUTS, TINY_ARCHITECTURE)
        assert model.layers[0].num_inputs == NUM_INPUTS
        assert model.layers[0].num_outputs == 16
        assert model.layers[1].num_inputs == 16
        assert model.layers[1].num_outputs == 8
        assert model.layers[2].num_inputs == 8
        assert model.layers[2].num_outputs == 4


class TestCreateLearner:
    def test_targets_correct_layer(self):
        from applications.deep_linear.model import create_model, create_learner

        model = create_model(SETUP, NUM_INPUTS, TINY_ARCHITECTURE)
        for i in range(3):
            learner = create_learner(model, i, SETUP)
            assert learner.layer is model.layers[i]

    def test_has_all_components(self):
        from applications.deep_linear.model import create_model, create_learner

        model = create_model(SETUP, NUM_INPUTS, TINY_ARCHITECTURE)
        learner = create_learner(model, 0, SETUP)
        assert learner.learning_mechanism is not None
        assert learner.competition is not None
        assert learner.threshold_adaptation is not None


class TestTrainLayerwise:
    def test_modifies_trained_layer_weights(self, train_loader, val_loader):
        from applications.deep_linear.model import create_model, train_layerwise

        torch.manual_seed(42)
        model = create_model(SETUP, NUM_INPUTS, TINY_ARCHITECTURE)
        weights_before = model.layers[0].weights.detach().clone()

        train_layerwise(
            model,
            SETUP,
            train_loader,
            val_loader,
            SPIKE_SHAPE,
            num_layers=1,
            num_epochs_per_layer=1,
        )

        assert not torch.equal(model.layers[0].weights, weights_before)

    def test_does_not_modify_untrained_layers(self, train_loader, val_loader):
        from applications.deep_linear.model import create_model, train_layerwise

        torch.manual_seed(42)
        model = create_model(SETUP, NUM_INPUTS, TINY_ARCHITECTURE)
        layer1_weights_before = model.layers[1].weights.detach().clone()
        layer2_weights_before = model.layers[2].weights.detach().clone()

        train_layerwise(
            model,
            SETUP,
            train_loader,
            val_loader,
            SPIKE_SHAPE,
            num_layers=1,
            num_epochs_per_layer=1,
        )

        assert torch.equal(model.layers[1].weights, layer1_weights_before)
        assert torch.equal(model.layers[2].weights, layer2_weights_before)


class TestApplyPba:
    def test_modifies_thresholds(self, train_loader, val_loader):
        from applications.deep_linear.model import create_model, apply_pba

        torch.manual_seed(42)
        model = create_model(SETUP, NUM_INPUTS, TINY_ARCHITECTURE)
        thresholds_before = model.layers[0].thresholds.detach().clone()

        apply_pba(
            model,
            train_loader,
            SPIKE_SHAPE,
            num_layers=1,
            pba_kwargs={
                "tau": 20.0,
                "learning_rate": 0.1,
                "min_threshold": 1.0,
                "max_threshold": 100.0,
            },
            num_epochs=1,
        )

        assert not torch.equal(model.layers[0].thresholds, thresholds_before)
