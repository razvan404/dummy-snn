import os

import numpy as np
import torch

from preprocessing import apply_difference_of_gaussians_filter, apply_latency_encoding
from spiking import convert_to_spikes
from .dataloader import Dataloader


class MnistSpikesDataloader(Dataloader):
    def __init__(self, path: str, split: str, patch_size: tuple[int, int] = None):
        if split not in ["train", "test"]:
            raise ValueError("Invalid split")

        self.inputs = (
            torch.from_numpy(
                np.fromfile(os.path.join(path, f"X_{split}_subset.bin"), dtype=np.uint8)
                .reshape(-1, 28, 28)
                .copy()
            ).float()
            / 255.0
        )

        self.outputs = torch.from_numpy(
            np.fromfile(
                os.path.join(path, f"y_{split}_subset.bin"), dtype=np.uint8
            ).copy()
        ).long()

        self.length = self.inputs.shape[0]
        if self.length != self.outputs.shape[0]:
            raise ValueError("Invalid shape of the data")

        self.patch_size = patch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        image = self.inputs[idx]
        label = self.outputs[idx]

        if self.patch_size is not None:
            # TODO: random patches can be taken
            pass

        dog_image = apply_difference_of_gaussians_filter(image)
        times = apply_latency_encoding(dog_image)
        spikes = convert_to_spikes(times)

        return spikes, label, times
