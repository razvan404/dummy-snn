import logging

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

from spiking.preprocessing.whitening_kernels import (
    fit_whitening_kernels,
    apply_whitening_kernels,
    compute_patch_mean,
    load_kernels,
)
from spiking.preprocessing.whitened_spike_encoding import encode_whitened_image
from spiking.preprocessing.latency_encoding import discretize_times


class Cifar10WhitenedDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        patch_size: int = 9,
        epsilon: float = 1e-2,
        rho: float = 0.15,
        n_patches: int = 1_000_000,
        num_bins: int = 16,
        kernels: torch.Tensor | None = None,
        mean: torch.Tensor | None = None,
        kernels_path: str | None = None,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"Invalid split: {split!r}, must be 'train' or 'test'")

        train = split == "train"
        dataset = torchvision.datasets.CIFAR10(root, train=train, download=True)

        # (N, 32, 32, 3) uint8 → (N, 3, 32, 32) float [0, 1]
        images = torch.from_numpy(dataset.data).float() / 255.0
        images = images.permute(0, 3, 1, 2)

        self.outputs = torch.tensor(dataset.targets, dtype=torch.long)

        # Load pre-computed kernels, reuse provided kernels, or fit from data
        if kernels_path is not None:
            logger.info("  Loading whitening kernels from %s...", kernels_path)
            self.kernels = load_kernels(kernels_path)
            self.mean = compute_patch_mean(
                images, patch_size=patch_size, n_patches=n_patches
            )
        elif kernels is not None:
            self.kernels = kernels
            self.mean = mean
        else:
            logger.info(
                "  Fitting whitening kernels (%s, %d patches)...", split, n_patches
            )
            self.kernels, self.mean = fit_whitening_kernels(
                images,
                patch_size=patch_size,
                n_patches=n_patches,
                epsilon=epsilon,
                rho=rho,
            )

        # Apply whitening, then per-sample encode as spikes (Falez 2020 Section IV-A)
        logger.info("  Whitening + encoding %d images (%s)...", len(images), split)
        whitened = apply_whitening_kernels(images, self.kernels, self.mean)

        self.all_times = torch.stack(
            [
                discretize_times(encode_whitened_image(whitened[i]), num_bins=num_bins)
                for i in range(len(whitened))
            ]
        )
        logger.info("  Done (%s).", split)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Return (2*C, H, W) of the spike-encoded images."""
        return tuple(self.all_times.shape[1:])

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx: int):
        return self.all_times[idx], self.outputs[idx]


def create_cifar10_whitened(
    patch_size: int = 9,
    epsilon: float = 1e-2,
    rho: float = 0.15,
    num_bins: int = 16,
    kernels_path: str | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train and test loaders for whitened CIFAR-10.

    Fits whitening kernels on training data and reuses for test,
    or loads pre-computed kernels from kernels_path.
    """
    train_dataset = Cifar10WhitenedDataset(
        "data",
        "train",
        patch_size=patch_size,
        epsilon=epsilon,
        rho=rho,
        num_bins=num_bins,
        kernels_path=kernels_path,
    )
    test_dataset = Cifar10WhitenedDataset(
        "data",
        "test",
        patch_size=patch_size,
        num_bins=num_bins,
        kernels=train_dataset.kernels,
        mean=train_dataset.mean,
    )
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False)
    return train_loader, test_loader
