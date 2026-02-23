import os

import numpy as np
import torch

from .base import SpikeEncodingDataset


class MnistSubsetDataset(SpikeEncodingDataset):
    def __init__(
        self,
        path: str,
        split: str,
        image_shape: tuple[int, int] = None,
    ):
        if split not in ["train", "test"]:
            raise ValueError("Invalid split")

        inputs = (
            torch.from_numpy(
                np.fromfile(os.path.join(path, f"X_{split}_subset.bin"), dtype=np.uint8)
                .reshape(-1, 28, 28)
                .copy()
            ).float()
            / 255.0
        )

        outputs = torch.from_numpy(
            np.fromfile(
                os.path.join(path, f"y_{split}_subset.bin"), dtype=np.uint8
            ).copy()
        )

        if inputs.shape[0] != outputs.shape[0]:
            raise ValueError("Invalid shape of the data")

        super().__init__(inputs, outputs, image_shape)
