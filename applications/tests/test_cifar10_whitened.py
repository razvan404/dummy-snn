import torch
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from applications.datasets.cifar10_whitened import (
    Cifar10WhitenedDataset,
    create_cifar10_whitened,
)


def make_fake_cifar_data(n=20):
    """Create fake CIFAR-10 data matching torchvision format."""
    data = np.random.randint(0, 256, (n, 32, 32, 3), dtype=np.uint8)
    targets = list(np.random.randint(0, 10, n))
    return data, targets


@pytest.fixture
def mock_cifar10():
    """Patch torchvision.datasets.CIFAR10 to avoid downloading."""
    data, targets = make_fake_cifar_data(n=50)
    with patch("torchvision.datasets.CIFAR10") as mock_cls:
        mock_dataset = MagicMock()
        mock_dataset.data = data
        mock_dataset.targets = targets
        mock_cls.return_value = mock_dataset
        yield mock_cls


class TestCifar10WhitenedDataset:
    def test_spike_times_shape(self, mock_cifar10):
        dataset = Cifar10WhitenedDataset("data", "train", patch_size=5)
        times, label = dataset[0]
        # 3 RGB channels → 6 spike channels (pos + neg)
        assert times.shape == (6, 32, 32)

    def test_length(self, mock_cifar10):
        dataset = Cifar10WhitenedDataset("data", "train", patch_size=5)
        assert len(dataset) == 50

    def test_labels_are_correct(self, mock_cifar10):
        dataset = Cifar10WhitenedDataset("data", "train", patch_size=5)
        _, label = dataset[0]
        assert label.dtype == torch.long

    def test_image_shape_property(self, mock_cifar10):
        dataset = Cifar10WhitenedDataset("data", "train", patch_size=5)
        assert dataset.image_shape == (6, 32, 32)

    def test_kernels_and_mean_exposed(self, mock_cifar10):
        dataset = Cifar10WhitenedDataset("data", "train", patch_size=5)
        assert dataset.kernels.shape == (3, 3, 5, 5)
        assert dataset.mean.shape[0] == 3 * 5 * 5

    def test_prefitted_kernels(self, mock_cifar10):
        train_ds = Cifar10WhitenedDataset("data", "train", patch_size=5)
        # Use train kernels for test set
        test_ds = Cifar10WhitenedDataset(
            "data",
            "test",
            patch_size=5,
            kernels=train_ds.kernels,
            mean=train_ds.mean,
        )
        assert test_ds.kernels is train_ds.kernels
        times, _ = test_ds[0]
        assert times.shape == (6, 32, 32)

    def test_spike_times_contain_inf(self, mock_cifar10):
        """Some pixels should not spike (inf)."""
        dataset = Cifar10WhitenedDataset("data", "train", patch_size=5)
        times, _ = dataset[0]
        assert torch.isinf(times).any()

    def test_finite_times_in_valid_range(self, mock_cifar10):
        dataset = Cifar10WhitenedDataset("data", "train", patch_size=5)
        times, _ = dataset[0]
        finite = times[torch.isfinite(times)]
        if len(finite) > 0:
            assert (finite >= 0).all()
            assert (finite < 1).all()

    def test_invalid_split_raises(self, mock_cifar10):
        with pytest.raises(ValueError, match="Invalid split"):
            Cifar10WhitenedDataset("data", "invalid", patch_size=5)


class TestCreateCifar10Whitened:
    def test_returns_loaders(self, mock_cifar10):
        train_loader, test_loader = create_cifar10_whitened(patch_size=5)
        assert train_loader is not None
        assert test_loader is not None

    def test_loader_yields_correct_shape(self, mock_cifar10):
        train_loader, _ = create_cifar10_whitened(patch_size=5)
        times, label = next(iter(train_loader))
        assert times.shape == (6, 32, 32)
