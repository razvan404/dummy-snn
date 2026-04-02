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


class TestPatchExtraction:
    def test_patch_output_shape(self):
        from applications.datasets.base import SpikeEncodingDataset

        inputs = torch.rand(5, 8, 8)
        outputs = torch.arange(5)
        ds = SpikeEncodingDataset(inputs, outputs, patch_size=3, num_patches=4)

        patches, label = ds[0]
        # 2 DoG channels, 3x3 patches, 4 patches
        assert patches.shape == (4, 2, 3, 3)
        assert label.dtype == torch.long

    def test_positions_within_bounds(self):
        from applications.datasets.base import SpikeEncodingDataset

        inputs = torch.rand(10, 8, 8)
        outputs = torch.arange(10)
        ds = SpikeEncodingDataset(inputs, outputs, patch_size=3, num_patches=5)

        positions = ds.patch_positions
        assert positions.shape == (10, 5, 2)
        # all_times has shape (N, 2, 8, 8), max valid row/col = 8 - 3 = 5
        assert (positions[:, :, 0] >= 0).all()
        assert (positions[:, :, 0] <= 5).all()
        assert (positions[:, :, 1] >= 0).all()
        assert (positions[:, :, 1] <= 5).all()

    def test_positions_distinct_per_image(self):
        from applications.datasets.base import SpikeEncodingDataset

        inputs = torch.rand(5, 10, 10)
        outputs = torch.arange(5)
        ds = SpikeEncodingDataset(inputs, outputs, patch_size=3, num_patches=8)

        positions = ds.patch_positions  # (5, 8, 2)
        for i in range(5):
            pos_tuples = set()
            for j in range(8):
                r, c = positions[i, j, 0].item(), positions[i, j, 1].item()
                pos_tuples.add((r, c))
            assert len(pos_tuples) == 8, f"Duplicate positions in image {i}"

    def test_no_patches_returns_full_image(self):
        from applications.datasets.base import SpikeEncodingDataset

        inputs = torch.rand(3, 8, 8)
        outputs = torch.tensor([0, 1, 2])
        ds = SpikeEncodingDataset(inputs, outputs)

        times, _ = ds[0]
        assert times.shape == (2, 8, 8)
        assert ds.patch_positions is None

    def test_cache_roundtrip(self, tmp_path):
        from applications.datasets.base import SpikeEncodingDataset

        inputs = torch.rand(5, 8, 8)
        outputs = torch.arange(5)
        cache = str(tmp_path / "times.pt")

        torch.manual_seed(42)
        ds1 = SpikeEncodingDataset(
            inputs, outputs, cache_path=cache, patch_size=3, num_patches=4
        )
        pos1 = ds1.patch_positions.clone()

        # Second load should read from cache regardless of seed
        torch.manual_seed(999)
        ds2 = SpikeEncodingDataset(
            inputs, outputs, cache_path=cache, patch_size=3, num_patches=4
        )
        torch.testing.assert_close(ds2.patch_positions, pos1)

    def test_different_num_patches_different_cache(self, tmp_path):
        from applications.datasets.base import SpikeEncodingDataset

        inputs = torch.rand(5, 8, 8)
        outputs = torch.arange(5)
        cache = str(tmp_path / "times.pt")

        ds_a = SpikeEncodingDataset(
            inputs, outputs, cache_path=cache, patch_size=3, num_patches=2
        )
        ds_b = SpikeEncodingDataset(
            inputs, outputs, cache_path=cache, patch_size=3, num_patches=4
        )
        # Different shapes — they use separate cache files
        assert ds_a.patch_positions.shape[1] == 2
        assert ds_b.patch_positions.shape[1] == 4

    def test_patch_size_5(self):
        from applications.datasets.base import SpikeEncodingDataset

        inputs = torch.rand(3, 12, 12)
        outputs = torch.tensor([0, 1, 2])
        ds = SpikeEncodingDataset(inputs, outputs, patch_size=5, num_patches=3)

        patches, _ = ds[0]
        assert patches.shape == (3, 2, 5, 5)


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


class TestFer2013Dataset:
    def _make_fake_kaggle_dir(self, tmp_path, split="train", n_per_class=2):
        """Create fake kagglehub directory with tiny JPEG images per class."""
        import numpy as np
        from PIL import Image

        classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        split_dir = tmp_path / split
        for cls_name in classes:
            cls_dir = split_dir / cls_name
            cls_dir.mkdir(parents=True)
            for i in range(n_per_class):
                img = Image.fromarray(
                    np.random.randint(0, 256, (48, 48), dtype=np.uint8), mode="L"
                )
                img.save(cls_dir / f"img_{i}.jpg")
        return str(tmp_path)

    @patch("applications.datasets.fer2013.kagglehub")
    def test_returns_times_label(self, mock_kagglehub, tmp_path):
        mock_kagglehub.dataset_download.return_value = self._make_fake_kaggle_dir(
            tmp_path
        )

        from applications.datasets.fer2013 import Fer2013Dataset

        ds = Fer2013Dataset(split="train")
        times, label = ds[0]

        assert isinstance(times, torch.Tensor)
        assert isinstance(label, torch.Tensor)

    @patch("applications.datasets.fer2013.kagglehub")
    def test_inputs_are_grayscale_float(self, mock_kagglehub, tmp_path):
        mock_kagglehub.dataset_download.return_value = self._make_fake_kaggle_dir(
            tmp_path
        )

        from applications.datasets.fer2013 import Fer2013Dataset

        ds = Fer2013Dataset(split="train")
        assert ds.inputs.ndim == 3  # (N, H, W)
        assert ds.inputs.dtype == torch.float32
        assert ds.inputs.min() >= 0.0
        assert ds.inputs.max() <= 1.0

    @patch("applications.datasets.fer2013.kagglehub")
    def test_native_image_shape(self, mock_kagglehub, tmp_path):
        mock_kagglehub.dataset_download.return_value = self._make_fake_kaggle_dir(
            tmp_path
        )

        from applications.datasets.fer2013 import Fer2013Dataset

        ds = Fer2013Dataset(split="train")
        assert ds.image_shape == (48, 48)

    @patch("applications.datasets.fer2013.kagglehub")
    def test_split_validation(self, mock_kagglehub):
        from applications.datasets.fer2013 import Fer2013Dataset

        with pytest.raises(ValueError, match="split"):
            Fer2013Dataset(split="val")

    @patch("applications.datasets.fer2013.kagglehub")
    def test_labels_match_sorted_class_dirs(self, mock_kagglehub, tmp_path):
        mock_kagglehub.dataset_download.return_value = self._make_fake_kaggle_dir(
            tmp_path, n_per_class=1
        )

        from applications.datasets.fer2013 import Fer2013Dataset

        ds = Fer2013Dataset(split="train")
        assert len(ds) == 7
        labels = [ds[i][1].item() for i in range(7)]
        assert labels == list(range(7))


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

    @patch("applications.datasets.fer2013.kagglehub")
    def test_fer2013_key_returns_loaders(self, mock_kagglehub, tmp_path):
        import numpy as np
        from PIL import Image
        from torch.utils.data import DataLoader

        classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        for split in ("train", "test"):
            split_dir = tmp_path / split
            for cls_name in classes:
                cls_dir = split_dir / cls_name
                cls_dir.mkdir(parents=True)
                img = Image.fromarray(
                    np.random.randint(0, 256, (48, 48), dtype=np.uint8), mode="L"
                )
                img.save(cls_dir / "img_0.jpg")
        mock_kagglehub.dataset_download.return_value = str(tmp_path)

        from applications.datasets import create_dataset

        train_loader, test_loader = create_dataset("fer2013")
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
