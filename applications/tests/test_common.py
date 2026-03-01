import os

import torch
import pytest

from applications.common import set_seed, evaluate_model
from applications.datasets import create_dataset


class TestSetSeed:
    def test_produces_deterministic_torch_randoms(self):
        set_seed(123)
        a = torch.rand(5)
        set_seed(123)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_different_seeds_produce_different_values(self):
        set_seed(1)
        a = torch.rand(5)
        set_seed(2)
        b = torch.rand(5)
        assert not torch.equal(a, b)


class TestEvaluateModel:
    def test_returns_train_and_val_metrics_with_expected_keys(self):
        from spiking.tests.test_evaluation import make_fake_dataloader, make_layer

        torch.manual_seed(42)
        shape = (2, 4, 4)
        num_inputs = 2 * 4 * 4
        layer = make_layer(num_inputs=num_inputs, num_outputs=5)
        train_loader = make_fake_dataloader(num_samples=10, shape=shape)
        val_loader = make_fake_dataloader(num_samples=5, shape=shape)

        train_metrics, val_metrics = evaluate_model(
            layer, train_loader, val_loader, image_shape=shape
        )

        for metrics in (train_metrics, val_metrics):
            assert isinstance(metrics, dict)
            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics


class TestCreateDataset:
    def test_raises_value_error_for_unknown_name(self):
        with pytest.raises(ValueError, match="unknown"):
            create_dataset("imagenet")

    @pytest.mark.skipif(
        not os.path.exists("data/mnist-subset"),
        reason="MNIST data not available",
    )
    def test_mnist_subset_returns_train_and_test_loaders(self):
        from torch.utils.data import DataLoader

        train_loader, test_loader = create_dataset("mnist_subset")
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
