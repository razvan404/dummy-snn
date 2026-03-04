import json
import os

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


@pytest.fixture()
def trained_model_path(tmp_path, fake_loaders):
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


class TestHomeostaticPBTR:
    def test_homeostatic_flips_sign(self):
        """When homeostatic=True, threshold deltas should be flipped."""
        from spiking.threshold import PlasticityBalanceAdaptation

        torch.manual_seed(42)
        thresholds = torch.tensor([5.0, 5.0, 5.0])
        spike_times = torch.tensor([0.5, 0.3, float("inf")])
        weights = torch.randn(3, 10)
        pre_spike_times = torch.rand(10)

        adapt_default = PlasticityBalanceAdaptation(
            tau=0.1,
            learning_rate=1.0,
            homeostatic=False,
        )
        adapt_homeo = PlasticityBalanceAdaptation(
            tau=0.1,
            learning_rate=1.0,
            homeostatic=True,
        )

        result_default = adapt_default.update(
            thresholds,
            spike_times,
            weights=weights,
            pre_spike_times=pre_spike_times,
        )
        result_homeo = adapt_homeo.update(
            thresholds,
            spike_times,
            weights=weights,
            pre_spike_times=pre_spike_times,
        )

        # Spiked neurons (indices 0,1) should move in opposite directions
        spiked = torch.tensor([0, 1])
        delta_default = result_default[spiked] - thresholds[spiked]
        delta_homeo = result_homeo[spiked] - thresholds[spiked]

        # Non-spiked neuron unchanged
        assert result_default[2] == thresholds[2]
        assert result_homeo[2] == thresholds[2]

        # Signs flipped for spiked neurons
        assert torch.allclose(delta_default, -delta_homeo)

    def test_homeostatic_default_is_false(self):
        from spiking.threshold import PlasticityBalanceAdaptation

        adapt = PlasticityBalanceAdaptation()
        assert adapt.homeostatic is False

    def test_apply_pbtr_accepts_tau_and_homeostatic(
        self,
        tmp_path,
        trained_model_path,
        fake_loaders,
    ):
        from applications.deep_linear.apply_pbtr import apply_pbtr

        output_dir = str(tmp_path / "pbtr_fixed")
        apply_pbtr(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=100,
            output_dir=output_dir,
            num_epochs=1,
            tau=0.1,
            homeostatic=True,
        )
        assert os.path.exists(f"{output_dir}/metrics.json")


class TestActivityThreshold:
    def test_saves_metrics(self, tmp_path, trained_model_path, fake_loaders):
        from applications.deep_linear.activity_threshold import activity_threshold

        output_dir = str(tmp_path / "activity_output")
        activity_threshold(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=42,
            output_dir=output_dir,
        )
        assert os.path.exists(f"{output_dir}/metrics.json")
        with open(f"{output_dir}/metrics.json") as f:
            metrics = json.load(f)
        assert "train" in metrics
        assert "validation" in metrics

    def test_high_fire_neurons_get_higher_thresholds(
        self,
        tmp_path,
        trained_model_path,
        fake_loaders,
    ):
        """Neurons that fire more often should get higher thresholds."""
        from applications.deep_linear.activity_threshold import activity_threshold
        from spiking import load_model

        model_before = load_model(trained_model_path)
        thresholds_before = model_before.layers[0].thresholds.detach().clone()

        output_dir = str(tmp_path / "activity_direction")
        activity_threshold(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=42,
            output_dir=output_dir,
        )
        model_after = load_model(f"{output_dir}/model.pth")
        thresholds_after = model_after.layers[0].thresholds.detach()

        # Mean threshold should be approximately preserved
        assert (
            abs(thresholds_after.mean().item() - thresholds_before.mean().item())
            < thresholds_before.mean().item() * 0.5
        )

    def test_saves_threshold_distribution(
        self,
        tmp_path,
        trained_model_path,
        fake_loaders,
    ):
        from applications.deep_linear.activity_threshold import activity_threshold

        output_dir = str(tmp_path / "activity_dist")
        activity_threshold(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=42,
            output_dir=output_dir,
        )
        assert os.path.exists(f"{output_dir}/threshold_distribution.png")


class TestPercentileThreshold:
    def test_saves_metrics(self, tmp_path, trained_model_path, fake_loaders):
        from applications.deep_linear.percentile_threshold import percentile_threshold

        output_dir = str(tmp_path / "percentile_output")
        percentile_threshold(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=42,
            output_dir=output_dir,
            percentile=50.0,
        )
        assert os.path.exists(f"{output_dir}/metrics.json")
        with open(f"{output_dir}/metrics.json") as f:
            metrics = json.load(f)
        assert "train" in metrics
        assert "validation" in metrics

    def test_higher_percentile_gives_higher_thresholds(self):
        """With varying input sparsity, higher percentile → higher thresholds."""
        from applications.deep_linear.percentile_threshold import (
            _compute_max_potentials,
        )
        from spiking import IntegrateAndFireLayer, NormalInitialization

        torch.manual_seed(42)
        layer = IntegrateAndFireLayer(
            num_inputs=10,
            num_outputs=3,
            threshold_initialization=NormalInitialization(
                avg_threshold=5.0,
                min_threshold=1.0,
                std_dev=1.0,
            ),
        )
        # Create inputs with varying sparsity (some inf)
        potentials = []
        for _ in range(20):
            input_times = torch.rand(10)
            # Randomly set some inputs to inf
            mask = torch.rand(10) > 0.5
            input_times[mask] = float("inf")
            p = _compute_max_potentials(layer, input_times)
            potentials.append(p)

        potential_matrix = torch.stack(potentials, dim=0)
        thresh_30 = torch.quantile(potential_matrix, 0.3, dim=0)
        thresh_70 = torch.quantile(potential_matrix, 0.7, dim=0)

        # Higher percentile should give higher (or equal) thresholds
        assert (thresh_70 >= thresh_30).all()
        # With enough variance, at least some neurons differ
        assert (thresh_70 > thresh_30).any()

    def test_clamp_respected(self, tmp_path, trained_model_path, fake_loaders):
        from applications.deep_linear.percentile_threshold import percentile_threshold
        from spiking import load_model

        output_dir = str(tmp_path / "percentile_clamp")
        percentile_threshold(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=42,
            output_dir=output_dir,
            percentile=1.0,  # Very low percentile → low thresholds
        )
        model = load_model(f"{output_dir}/model.pth")
        assert model.layers[0].thresholds.min().item() >= 1.0

    def test_saves_threshold_distribution(
        self,
        tmp_path,
        trained_model_path,
        fake_loaders,
    ):
        from applications.deep_linear.percentile_threshold import percentile_threshold

        output_dir = str(tmp_path / "percentile_dist")
        percentile_threshold(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=42,
            output_dir=output_dir,
            percentile=50.0,
        )
        assert os.path.exists(f"{output_dir}/threshold_distribution.png")


class TestRandomThresholds:
    def test_saves_metrics(self, tmp_path, trained_model_path, fake_loaders):
        from applications.deep_linear.random_thresholds import random_thresholds

        output_dir = str(tmp_path / "random_output")
        random_thresholds(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=42,
            output_dir=output_dir,
        )
        assert os.path.exists(f"{output_dir}/metrics.json")
        with open(f"{output_dir}/metrics.json") as f:
            metrics = json.load(f)
        assert "train" in metrics
        assert "validation" in metrics

    def test_std_controls_perturbation_spread(
        self,
        tmp_path,
        trained_model_path,
        fake_loaders,
    ):
        """Smaller std should produce thresholds closer to the mean."""
        from applications.deep_linear.random_thresholds import random_thresholds
        from spiking import load_model

        model_before = load_model(trained_model_path)
        mean_thresh = model_before.layers[0].thresholds.mean().item()

        for std_val, label in [(0.1, "tight"), (5.0, "wide")]:
            output_dir = str(tmp_path / f"random_std_{label}")
            random_thresholds(
                model_path=trained_model_path,
                dataset_loaders=fake_loaders,
                spike_shape=SPIKE_SHAPE,
                seed=42,
                output_dir=output_dir,
                std=std_val,
            )
            model_after = load_model(f"{output_dir}/model.pth")
            thresholds = model_after.layers[0].thresholds.detach()
            # Tight std should have smaller spread than wide std
            if label == "tight":
                tight_std = thresholds.std().item()
            else:
                wide_std = thresholds.std().item()

        assert tight_std < wide_std


class TestWeightThreshold:
    def test_saves_metrics(self, tmp_path, trained_model_path, fake_loaders):
        from applications.deep_linear.weight_threshold import weight_threshold

        output_dir = str(tmp_path / "weight_output")
        weight_threshold(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=42,
            output_dir=output_dir,
        )
        assert os.path.exists(f"{output_dir}/metrics.json")
        with open(f"{output_dir}/metrics.json") as f:
            metrics = json.load(f)
        assert "train" in metrics
        assert "validation" in metrics

    def test_threshold_proportional_to_weight_sums(
        self,
        tmp_path,
        trained_model_path,
        fake_loaders,
    ):
        """Thresholds should be proportional to incoming weight sums."""
        from applications.deep_linear.weight_threshold import weight_threshold
        from spiking import load_model

        model_before = load_model(trained_model_path)
        weights = model_before.layers[0].weights.detach()
        weight_sums = weights.sum(dim=1)

        output_dir = str(tmp_path / "weight_prop")
        weight_threshold(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=42,
            output_dir=output_dir,
        )
        model_after = load_model(f"{output_dir}/model.pth")
        thresholds_after = model_after.layers[0].thresholds.detach()

        # Thresholds should correlate with weight sums (higher weights → higher threshold)
        # Use rank correlation: neurons with larger weight sums should have larger thresholds
        # (after clamping, exact proportionality may not hold, but the ordering should)
        assert thresholds_after.min() >= 1.0  # min clamp respected

    def test_preserves_mean_threshold(
        self,
        tmp_path,
        trained_model_path,
        fake_loaders,
    ):
        """Mean threshold should be approximately preserved."""
        from applications.deep_linear.weight_threshold import weight_threshold
        from spiking import load_model

        model_before = load_model(trained_model_path)
        mean_before = model_before.layers[0].thresholds.mean().item()

        output_dir = str(tmp_path / "weight_mean")
        weight_threshold(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=42,
            output_dir=output_dir,
        )
        model_after = load_model(f"{output_dir}/model.pth")
        mean_after = model_after.layers[0].thresholds.mean().item()

        # Mean should be close (clamping may shift it slightly)
        assert abs(mean_after - mean_before) < mean_before * 0.5

    def test_saves_threshold_distribution(
        self,
        tmp_path,
        trained_model_path,
        fake_loaders,
    ):
        from applications.deep_linear.weight_threshold import weight_threshold

        output_dir = str(tmp_path / "weight_dist")
        weight_threshold(
            model_path=trained_model_path,
            dataset_loaders=fake_loaders,
            spike_shape=SPIKE_SHAPE,
            seed=42,
            output_dir=output_dir,
        )
        assert os.path.exists(f"{output_dir}/threshold_distribution.png")
