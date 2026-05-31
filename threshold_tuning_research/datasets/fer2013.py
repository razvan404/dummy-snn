from pathlib import Path

import kagglehub
import numpy as np
import torch
from PIL import Image

from .base import SpikeEncodingDataset


class Fer2013Dataset(SpikeEncodingDataset):
    def __init__(
        self,
        split: str,
        image_shape: tuple[int, int] | None = None,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"Invalid split: {split!r}, must be 'train' or 'test'")

        data_path = Path(kagglehub.dataset_download("msambare/fer2013"))
        split_dir = data_path / split

        images = []
        labels = []
        for label, cls_dir in enumerate(sorted(split_dir.iterdir())):
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.glob("*.jpg")):
                img = Image.open(img_path).convert("L")
                tensor = torch.from_numpy(np.array(img)).float() / 255.0
                images.append(tensor)
                labels.append(label)

        inputs = torch.stack(images)
        outputs = torch.tensor(labels)

        super().__init__(inputs, outputs, image_shape)
