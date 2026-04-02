import json
import os
from unittest.mock import patch

import torch
import pytest
from torch.utils.data import DataLoader

from spiking.tests.test_evaluation import FakeDataset

SPIKE_SHAPE = (2, 4, 4)
TINY_ARCHITECTURE = [8, 4, 2]


@pytest.fixture(scope="module")
def fake_loaders():
    torch.manual_seed(0)
    train_ds = FakeDataset(num_samples=10, shape=SPIKE_SHAPE)
    val_ds = FakeDataset(num_samples=5, shape=SPIKE_SHAPE)
    train_loader = DataLoader(train_ds, batch_size=None, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=None, shuffle=False)
    return train_loader, val_loader


class TestTrainLayer:
    def test_saves_model_file(self, tmp_path, fake_loaders):
        from applications.deep_linear.train import train_layer

        train_layer(
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=1,
            avg_threshold=5.0,
            output_dir=str(tmp_path),
            num_epochs=1,
            architecture=TINY_ARCHITECTURE,
        )
        assert (tmp_path / "model.pth").exists()

    def test_saves_metrics_json(self, tmp_path, fake_loaders):
        from applications.deep_linear.train import train_layer

        train_layer(
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=1,
            avg_threshold=5.0,
            output_dir=str(tmp_path),
            num_epochs=1,
            architecture=TINY_ARCHITECTURE,
        )
        with open(tmp_path / "metrics.json") as f:
            metrics = json.load(f)
        assert "train" in metrics
        assert "validation" in metrics
        assert "accuracy" in metrics["train"]

    def test_saves_setup_json(self, tmp_path, fake_loaders):
        from applications.deep_linear.train import train_layer

        train_layer(
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=1,
            avg_threshold=5.0,
            output_dir=str(tmp_path),
            num_epochs=1,
            architecture=TINY_ARCHITECTURE,
        )
        with open(tmp_path / "setup.json") as f:
            setup = json.load(f)
        assert setup["avg_threshold"] == 5.0
        assert setup["layer_idx"] == 0

    def test_saves_weights_figure_for_layer_0(self, tmp_path, fake_loaders):
        from applications.deep_linear.train import train_layer

        train_layer(
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=1,
            avg_threshold=5.0,
            output_dir=str(tmp_path),
            num_epochs=1,
            architecture=TINY_ARCHITECTURE,
        )
        assert (tmp_path / "weights.png").exists()

    def test_with_t_objective(self, tmp_path, fake_loaders):
        from applications.deep_linear.train import train_layer

        train_layer(
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=1,
            avg_threshold=5.0,
            output_dir=str(tmp_path),
            num_epochs=1,
            t_objective=0.5,
            architecture=TINY_ARCHITECTURE,
        )
        with open(tmp_path / "setup.json") as f:
            setup = json.load(f)
        assert setup["t_objective"] == 0.5
        assert (tmp_path / "model.pth").exists()

    def test_model_is_multilayer(self, tmp_path, fake_loaders):
        from applications.deep_linear.train import train_layer
        from spiking import load_model
        from spiking.layers import IntegrateAndFireMultilayer

        train_layer(
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=1,
            avg_threshold=5.0,
            output_dir=str(tmp_path),
            num_epochs=1,
            architecture=TINY_ARCHITECTURE,
        )
        model = load_model(str(tmp_path / "model.pth"))
        assert isinstance(model, IntegrateAndFireMultilayer)

    def test_saves_winner_counts_and_threshold_distribution(
        self, tmp_path, fake_loaders
    ):
        from applications.deep_linear.train import train_layer

        train_layer(
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=1,
            avg_threshold=5.0,
            output_dir=str(tmp_path),
            num_epochs=1,
            architecture=TINY_ARCHITECTURE,
        )
        assert (tmp_path / "winner_counts.json").exists()
        assert (tmp_path / "winner_counts.png").exists()
        assert (tmp_path / "threshold_distribution.png").exists()

        with open(tmp_path / "winner_counts.json") as f:
            counts = json.load(f)
        assert isinstance(counts, list)
        assert len(counts) == TINY_ARCHITECTURE[0]

    def test_trains_sub_model_for_layer_0(self, tmp_path, fake_loaders):
        """train_layer should pass a sub-model (not the full model) to train()."""
        from applications.deep_linear.train import train_layer

        captured = {}

        def capture_train(model, *args, **kwargs):
            captured["model"] = model

        with patch("applications.deep_linear.train.train", side_effect=capture_train):
            with patch(
                "applications.deep_linear.train.evaluate_model", return_value=({}, {})
            ):
                train_layer(
                    dataset_loaders=fake_loaders,
                    spike_shape=SPIKE_SHAPE,
                    seed=1,
                    avg_threshold=5.0,
                    output_dir=str(tmp_path),
                    num_epochs=1,
                    architecture=TINY_ARCHITECTURE,
                    layer_idx=0,
                )

        assert len(captured["model"].layers) == 1

    def test_evaluates_sub_model_for_layer_0(self, tmp_path, fake_loaders):
        """train_layer should pass a sub-model to evaluate_model()."""
        from applications.deep_linear.train import train_layer

        captured = {}

        def capture_eval(model, *args, **kwargs):
            captured["model"] = model
            return {"accuracy": 0.0}, {"accuracy": 0.0}

        with patch("applications.deep_linear.train.train"):
            with patch(
                "applications.deep_linear.train.evaluate_model",
                side_effect=capture_eval,
            ):
                train_layer(
                    dataset_loaders=fake_loaders,
                    spike_shape=SPIKE_SHAPE,
                    seed=1,
                    avg_threshold=5.0,
                    output_dir=str(tmp_path),
                    num_epochs=1,
                    architecture=TINY_ARCHITECTURE,
                    layer_idx=0,
                )

        assert len(captured["model"].layers) == 1


class TestApplyPbtr:
    @pytest.fixture()
    def trained_model_path(self, tmp_path, fake_loaders):
        from applications.deep_linear.train import train_layer

        train_layer(
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=1,
            avg_threshold=5.0,
            output_dir=str(tmp_path / "trained"),
            num_epochs=1,
            architecture=TINY_ARCHITECTURE,
        )
        return str(tmp_path / "trained" / "model.pth")

    def test_saves_model_and_metrics(self, tmp_path, trained_model_path, fake_loaders):
        from applications.deep_linear.apply_pbtr import apply_pbtr

        output_dir = str(tmp_path / "pbtr_output")
        apply_pbtr(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=100,
            output_dir=output_dir,
            num_epochs=1,
        )
        assert os.path.exists(f"{output_dir}/model.pth")
        assert os.path.exists(f"{output_dir}/metrics.json")

    def test_modifies_thresholds(self, tmp_path, trained_model_path, fake_loaders):
        from applications.deep_linear.apply_pbtr import apply_pbtr
        from spiking import load_model

        model_before = load_model(trained_model_path)
        thresholds_before = model_before.layers[0].thresholds.detach().clone()

        output_dir = str(tmp_path / "pbtr_output")
        apply_pbtr(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=100,
            output_dir=output_dir,
            num_epochs=1,
        )
        model_after = load_model(f"{output_dir}/model.pth")
        assert not torch.equal(model_after.layers[0].thresholds, thresholds_before)

    def test_does_not_save_weights_figure(
        self, tmp_path, trained_model_path, fake_loaders
    ):
        from applications.deep_linear.apply_pbtr import apply_pbtr

        output_dir = str(tmp_path / "pbtr_output")
        apply_pbtr(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=100,
            output_dir=output_dir,
            num_epochs=1,
        )
        assert not os.path.exists(f"{output_dir}/weights.png")

    def test_saves_threshold_distribution(
        self, tmp_path, trained_model_path, fake_loaders
    ):
        from applications.deep_linear.apply_pbtr import apply_pbtr

        output_dir = str(tmp_path / "pbtr_output")
        apply_pbtr(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=100,
            output_dir=output_dir,
            num_epochs=1,
        )
        assert os.path.exists(f"{output_dir}/threshold_distribution.png")

    def test_trains_sub_model_for_layer_0(
        self, tmp_path, trained_model_path, fake_loaders
    ):
        """apply_pbtr should pass a sub-model (not the full model) to train()."""
        from applications.deep_linear.apply_pbtr import apply_pbtr

        captured = {}

        def capture_train(model, *args, **kwargs):
            captured["model"] = model

        with patch(
            "applications.deep_linear.apply_pbtr.train", side_effect=capture_train
        ):
            with patch(
                "applications.deep_linear.apply_pbtr.evaluate_model",
                return_value=({}, {}),
            ):
                apply_pbtr(
                    model_path=trained_model_path,
                    dataset_loaders=fake_loaders,
                    spike_shape=SPIKE_SHAPE,
                    seed=100,
                    output_dir=str(tmp_path / "pbtr_sub"),
                    num_epochs=1,
                )

        assert len(captured["model"].layers) == 1

    def test_evaluates_sub_model_for_layer_0(
        self, tmp_path, trained_model_path, fake_loaders
    ):
        """apply_pbtr should pass a sub-model to evaluate_model()."""
        from applications.deep_linear.apply_pbtr import apply_pbtr

        captured = {}

        def capture_eval(model, *args, **kwargs):
            captured["model"] = model
            return {"accuracy": 0.0}, {"accuracy": 0.0}

        with patch("applications.deep_linear.apply_pbtr.train"):
            with patch(
                "applications.deep_linear.apply_pbtr.evaluate_model",
                side_effect=capture_eval,
            ):
                apply_pbtr(
                    model_path=trained_model_path,
                    dataset_loaders=fake_loaders,
                    spike_shape=SPIKE_SHAPE,
                    seed=100,
                    output_dir=str(tmp_path / "pbtr_sub_eval"),
                    num_epochs=1,
                )

        assert len(captured["model"].layers) == 1


class TestRandomThresholds:
    @pytest.fixture()
    def trained_model_path(self, tmp_path, fake_loaders):
        from applications.deep_linear.train import train_layer

        train_layer(
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=1,
            avg_threshold=5.0,
            output_dir=str(tmp_path / "trained"),
            num_epochs=1,
            architecture=TINY_ARCHITECTURE,
        )
        return str(tmp_path / "trained" / "model.pth")

    def test_saves_metrics(self, tmp_path, trained_model_path, fake_loaders):
        from applications.deep_linear.random_thresholds import random_thresholds

        output_dir = str(tmp_path / "random_output")
        random_thresholds(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=200,
            output_dir=output_dir,
        )
        assert os.path.exists(f"{output_dir}/metrics.json")
        with open(f"{output_dir}/metrics.json") as f:
            metrics = json.load(f)
        assert "train" in metrics
        assert "validation" in metrics

    def test_saves_model(self, tmp_path, trained_model_path, fake_loaders):
        from applications.deep_linear.random_thresholds import random_thresholds

        output_dir = str(tmp_path / "random_output")
        random_thresholds(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=200,
            output_dir=output_dir,
        )
        assert os.path.exists(f"{output_dir}/model.pth")


class TestSaveWeightFigure:
    def test_creates_png_file(self, tmp_path):
        from applications.deep_linear.visualize_weights import save_weight_figure
        from spiking import IntegrateAndFireLayer, NormalInitialization

        torch.manual_seed(42)
        layer = IntegrateAndFireLayer(
            num_inputs=32,
            num_outputs=8,
            threshold_initialization=NormalInitialization(5.0, 1.0, 0.5),
        )
        save_weight_figure(layer, (2, 4, 4), str(tmp_path / "weights.png"))
        assert (tmp_path / "weights.png").exists()

    def test_file_is_valid_png(self, tmp_path):
        from applications.deep_linear.visualize_weights import save_weight_figure
        from spiking import IntegrateAndFireLayer, NormalInitialization

        torch.manual_seed(42)
        layer = IntegrateAndFireLayer(
            num_inputs=32,
            num_outputs=8,
            threshold_initialization=NormalInitialization(5.0, 1.0, 0.5),
        )
        path = str(tmp_path / "weights.png")
        save_weight_figure(layer, (2, 4, 4), path)
        with open(path, "rb") as f:
            assert f.read(4) == b"\x89PNG"
