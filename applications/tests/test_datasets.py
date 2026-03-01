import torch
import pytest
from unittest.mock import patch


class TestSpikeEncodingDataset:
    def test_getitem_returns_times_label(self):
        from applications.datasets.base import SpikeEncodingDataset

        inputs = torch.rand(3, 8, 8)
        outputs = torch.tensor([0, 1, 2])
        ds = SpikeEncodingDataset(inputs, outputs)

        times, label = ds[0]

        assert isinstance(times, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long

    def test_len(self):
        from applications.datasets.base import SpikeEncodingDataset

        inputs = torch.rand(5, 8, 8)
        outputs = torch.arange(5)
        ds = SpikeEncodingDataset(inputs, outputs)
        assert len(ds) == 5

    def test_times_has_two_channels(self):
        """DoG produces 2 channels (on/off) from a single grayscale image."""
        from applications.datasets.base import SpikeEncodingDataset

        inputs = torch.rand(2, 8, 8)
        outputs = torch.tensor([0, 1])
        ds = SpikeEncodingDataset(inputs, outputs)

        times, _ = ds[0]
        assert times.shape[0] == 2  # on/off channels

    def test_resize_when_image_shape_given(self):
        from applications.datasets.base import SpikeEncodingDataset

        inputs = torch.rand(2, 28, 28)
        outputs = torch.tensor([0, 1])
        ds = SpikeEncodingDataset(inputs, outputs, image_shape=(14, 14))

        times, _ = ds[0]
        # DoG produces 2 channels; spatial dims should be 14x14
        assert times.shape == (2, 14, 14)

    def test_no_resize_when_image_shape_none(self):
        from applications.datasets.base import SpikeEncodingDataset

        inputs = torch.rand(2, 10, 10)
        outputs = torch.tensor([0, 1])
        ds = SpikeEncodingDataset(inputs, outputs)

        times, _ = ds[0]
        assert times.shape == (2, 10, 10)

    def test_image_shape_returns_hw_tuple(self):
        from applications.datasets.base import SpikeEncodingDataset

        inputs = torch.rand(3, 12, 10)
        outputs = torch.tensor([0, 1, 2])
        ds = SpikeEncodingDataset(inputs, outputs)
        assert ds.image_shape == (12, 10)

    def test_image_shape_after_resize(self):
        from applications.datasets.base import SpikeEncodingDataset

        inputs = torch.rand(2, 28, 28)
        outputs = torch.tensor([0, 1])
        ds = SpikeEncodingDataset(inputs, outputs, image_shape=(14, 14))
        assert ds.image_shape == (14, 14)


class TestMnistDataset:
    def _make_fake_mnist_data(self):
        """Return fake (data, targets) tensors matching torchvision MNIST format."""
        data = torch.randint(0, 256, (10, 28, 28), dtype=torch.uint8)
        targets = torch.randint(0, 10, (10,))
        return data, targets

    @patch("torchvision.datasets.MNIST")
    def test_returns_times_label(self, mock_mnist_cls):
        data, targets = self._make_fake_mnist_data()
        mock_mnist_cls.return_value.data = data
        mock_mnist_cls.return_value.targets = targets

        from applications.datasets.mnist import MnistDataset

        ds = MnistDataset(root="data", split="train")
        times, label = ds[0]

        assert isinstance(times, torch.Tensor)
        assert isinstance(label, torch.Tensor)

    @patch("torchvision.datasets.MNIST")
    def test_inputs_are_float_0_to_1(self, mock_mnist_cls):
        data, targets = self._make_fake_mnist_data()
        mock_mnist_cls.return_value.data = data
        mock_mnist_cls.return_value.targets = targets

        from applications.datasets.mnist import MnistDataset

        ds = MnistDataset(root="data", split="train")
        assert ds.inputs.dtype == torch.float32
        assert ds.inputs.min() >= 0.0
        assert ds.inputs.max() <= 1.0

    @patch("torchvision.datasets.MNIST")
    def test_split_test(self, mock_mnist_cls):
        data, targets = self._make_fake_mnist_data()
        mock_mnist_cls.return_value.data = data
        mock_mnist_cls.return_value.targets = targets

        from applications.datasets.mnist import MnistDataset

        ds = MnistDataset(root="data", split="test")
        mock_mnist_cls.assert_called_with("data", train=False, download=True)

    @patch("torchvision.datasets.MNIST")
    def test_split_train(self, mock_mnist_cls):
        data, targets = self._make_fake_mnist_data()
        mock_mnist_cls.return_value.data = data
        mock_mnist_cls.return_value.targets = targets

        from applications.datasets.mnist import MnistDataset

        ds = MnistDataset(root="data", split="train")
        mock_mnist_cls.assert_called_with("data", train=True, download=True)

    @patch("torchvision.datasets.MNIST")
    def test_invalid_split_raises(self, mock_mnist_cls):
        from applications.datasets.mnist import MnistDataset

        with pytest.raises(ValueError, match="split"):
            MnistDataset(root="data", split="val")


class TestFashionMnistDataset:
    @patch("torchvision.datasets.FashionMNIST")
    def test_returns_times_label(self, mock_cls):
        mock_cls.return_value.data = torch.randint(
            0, 256, (5, 28, 28), dtype=torch.uint8
        )
        mock_cls.return_value.targets = torch.randint(0, 10, (5,))

        from applications.datasets.fashion_mnist import FashionMnistDataset

        ds = FashionMnistDataset(root="data", split="train")
        times, label = ds[0]

        assert isinstance(times, torch.Tensor)
        assert isinstance(label, torch.Tensor)


class TestCifar10Dataset:
    def _make_fake_cifar_data(self):
        # CIFAR-10 data is numpy array (N, 32, 32, 3)
        import numpy as np

        data = np.random.randint(0, 256, (5, 32, 32, 3), dtype=np.uint8)
        targets = list(range(5))
        return data, targets

    @patch("torchvision.datasets.CIFAR10")
    def test_returns_times_label(self, mock_cls):
        data, targets = self._make_fake_cifar_data()
        mock_cls.return_value.data = data
        mock_cls.return_value.targets = targets

        from applications.datasets.cifar10 import Cifar10Dataset

        ds = Cifar10Dataset(root="data", split="train")
        times, label = ds[0]

        assert isinstance(times, torch.Tensor)
        assert isinstance(label, torch.Tensor)

    @patch("torchvision.datasets.CIFAR10")
    def test_inputs_are_single_channel_grayscale(self, mock_cls):
        data, targets = self._make_fake_cifar_data()
        mock_cls.return_value.data = data
        mock_cls.return_value.targets = targets

        from applications.datasets.cifar10 import Cifar10Dataset

        ds = Cifar10Dataset(root="data", split="train")
        # inputs should be (N, H, W) — single channel
        assert ds.inputs.ndim == 3
        assert ds.inputs.dtype == torch.float32
        assert ds.inputs.min() >= 0.0
        assert ds.inputs.max() <= 1.0

    @patch("torchvision.datasets.CIFAR10")
    def test_image_shape_resize(self, mock_cls):
        data, targets = self._make_fake_cifar_data()
        mock_cls.return_value.data = data
        mock_cls.return_value.targets = targets

        from applications.datasets.cifar10 import Cifar10Dataset

        ds = Cifar10Dataset(root="data", split="train", image_shape=(16, 16))
        times, _ = ds[0]
        assert times.shape == (2, 16, 16)


class TestCreateDatasetFactory:
    def test_raises_for_unknown_name(self):
        from applications.datasets import create_dataset

        with pytest.raises(ValueError, match="unknown"):
            create_dataset("imagenet")

    @patch("torchvision.datasets.MNIST")
    def test_mnist_key_returns_loaders(self, mock_cls):
        from torch.utils.data import DataLoader

        mock_cls.return_value.data = torch.randint(
            0, 256, (5, 28, 28), dtype=torch.uint8
        )
        mock_cls.return_value.targets = torch.randint(0, 10, (5,))

        from applications.datasets import create_dataset

        train_loader, test_loader = create_dataset("mnist")
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    @patch("torchvision.datasets.MNIST")
    def test_mnist_dataset_exposes_image_shape(self, mock_cls):
        mock_cls.return_value.data = torch.randint(
            0, 256, (5, 28, 28), dtype=torch.uint8
        )
        mock_cls.return_value.targets = torch.randint(0, 10, (5,))

        from applications.datasets import create_dataset

        train_loader, _ = create_dataset("mnist")
        assert train_loader.dataset.image_shape == (28, 28)

    @patch("torchvision.datasets.FashionMNIST")
    def test_fashion_mnist_key_returns_loaders(self, mock_cls):
        from torch.utils.data import DataLoader

        mock_cls.return_value.data = torch.randint(
            0, 256, (5, 28, 28), dtype=torch.uint8
        )
        mock_cls.return_value.targets = torch.randint(0, 10, (5,))

        from applications.datasets import create_dataset

        train_loader, test_loader = create_dataset("fashion_mnist")
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    @patch("torchvision.datasets.CIFAR10")
    def test_cifar10_key_returns_loaders(self, mock_cls):
        import numpy as np
        from torch.utils.data import DataLoader

        mock_cls.return_value.data = np.random.randint(
            0, 256, (5, 32, 32, 3), dtype=np.uint8
        )
        mock_cls.return_value.targets = list(range(5))

        from applications.datasets import create_dataset

        train_loader, test_loader = create_dataset("cifar10")
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    @patch("torchvision.datasets.CIFAR10")
    def test_cifar10_native_shape(self, mock_cls):
        import numpy as np

        mock_cls.return_value.data = np.random.randint(
            0, 256, (5, 32, 32, 3), dtype=np.uint8
        )
        mock_cls.return_value.targets = list(range(5))

        from applications.datasets import create_dataset

        train_loader, _ = create_dataset("cifar10")
        assert train_loader.dataset.image_shape == (32, 32)
