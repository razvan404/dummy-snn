import torch
import torchvision

from .base import SpikeEncodingDataset


class Cifar10Dataset(SpikeEncodingDataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_shape: tuple[int, int] | None = None,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"Invalid split: {split!r}, must be 'train' or 'test'")

        train = split == "train"
        dataset = torchvision.datasets.CIFAR10(root, train=train, download=True)

        # dataset.data is numpy (N, 32, 32, 3) uint8
        data = torch.from_numpy(dataset.data).float() / 255.0
        # RGB → grayscale: (N, H, W, 3) → (N, H, W)
        inputs = (data * torch.tensor([0.2989, 0.5870, 0.1140])).sum(dim=-1)

        outputs = torch.tensor(dataset.targets)

        super().__init__(inputs, outputs, image_shape)
