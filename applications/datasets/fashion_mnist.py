import torchvision

from .base import SpikeEncodingDataset


class FashionMnistDataset(SpikeEncodingDataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_shape: tuple[int, int] | None = None,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"Invalid split: {split!r}, must be 'train' or 'test'")

        train = split == "train"
        dataset = torchvision.datasets.FashionMNIST(root, train=train, download=True)

        inputs = dataset.data.float() / 255.0
        outputs = dataset.targets

        super().__init__(inputs, outputs, image_shape)
