import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from spiking.preprocessing.whitening_kernels import (
    fit_whitening_kernels,
    apply_whitening_kernels,
)
from spiking.preprocessing.whitened_spike_encoding import encode_whitened_image


class Cifar10WhitenedDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        patch_size: int = 9,
        epsilon: float = 1e-2,
        rho: float = 1.0,
        n_patches: int = 1_000_000,
        kernels: torch.Tensor | None = None,
        mean: torch.Tensor | None = None,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"Invalid split: {split!r}, must be 'train' or 'test'")

        train = split == "train"
        dataset = torchvision.datasets.CIFAR10(root, train=train, download=True)

        # (N, 32, 32, 3) uint8 → (N, 3, 32, 32) float [0, 1]
        images = torch.from_numpy(dataset.data).float() / 255.0
        images = images.permute(0, 3, 1, 2)

        self.outputs = torch.tensor(dataset.targets, dtype=torch.long)

        # Fit or reuse whitening kernels
        if kernels is None:
            self.kernels, self.mean = fit_whitening_kernels(
                images,
                patch_size=patch_size,
                n_patches=n_patches,
                epsilon=epsilon,
                rho=rho,
            )
        else:
            self.kernels = kernels
            self.mean = mean

        # Apply whitening and encode as spikes
        whitened = apply_whitening_kernels(images, self.kernels, self.mean)

        self.all_times = torch.stack(
            [encode_whitened_image(whitened[i]) for i in range(len(whitened))]
        )

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
    rho: float = 1.0,
) -> tuple[DataLoader, DataLoader]:
    """Create train and test loaders for whitened CIFAR-10.

    Fits whitening kernels on training data and reuses for test.
    """
    train_dataset = Cifar10WhitenedDataset(
        "data",
        "train",
        patch_size=patch_size,
        epsilon=epsilon,
        rho=rho,
    )
    test_dataset = Cifar10WhitenedDataset(
        "data",
        "test",
        patch_size=patch_size,
        kernels=train_dataset.kernels,
        mean=train_dataset.mean,
    )
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False)
    return train_loader, test_loader
