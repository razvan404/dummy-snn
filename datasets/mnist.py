import os

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing import apply_difference_of_gaussians_filter, apply_latency_encoding
from spiking import convert_to_spikes


class MnistDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str,
        patch_size: tuple[int, int] = None,
        image_shape: tuple[int, int] = None,
    ):
        if split not in ["train", "test"]:
            raise ValueError("Invalid split")

        self.patch_size = patch_size
        self.image_shape = image_shape

        self.inputs = self._process_inputs(os.path.join(path, f"X_{split}_subset.bin"))
        self.outputs = torch.from_numpy(
            np.fromfile(
                os.path.join(path, f"y_{split}_subset.bin"), dtype=np.uint8
            ).copy()
        ).long()

        self.length = self.inputs.shape[0]
        if self.length != self.outputs.shape[0]:
            raise ValueError("Invalid shape of the data")

    def _process_inputs(self, path: str = None):
        inputs = (
            torch.from_numpy(
                np.fromfile(path, dtype=np.uint8).reshape(-1, 28, 28).copy()
            ).float()
            / 255.0
        )

        if self.patch_size is not None:
            raise NotImplementedError("Patch size processing is not implemented yet")

        if self.image_shape is not None:
            inputs = torch.nn.functional.interpolate(
                inputs.unsqueeze(0),
                size=self.image_shape,
                mode="nearest",
            ).squeeze(0)

        return inputs

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        image = self.inputs[idx]
        label = self.outputs[idx]

        dog_image = apply_difference_of_gaussians_filter(image)
        times = apply_latency_encoding(dog_image)
        spikes = convert_to_spikes(times)

        # TODO: later
        # NC x H x W
        # (NC * NUM_PATCHES) x PATCH_H x PATCH_W
        # choose patches normally distributed

        return spikes, label, times
