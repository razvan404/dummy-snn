import torch
import torchvision

from .base import SpikeEncodingDataset


class Fer2013Dataset(SpikeEncodingDataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_shape: tuple[int, int] | None = None,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"Invalid split: {split!r}, must be 'train' or 'test'")

        dataset = torchvision.datasets.FER2013(root, split=split)

        inputs = torch.stack([img for img, _ in dataset._samples]).float() / 255.0
        outputs = torch.tensor([label for _, label in dataset._samples])

        super().__init__(inputs, outputs, image_shape)
