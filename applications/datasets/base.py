import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from spiking.preprocessing import (
    apply_difference_of_gaussians_filter_batch,
    apply_latency_encoding,
    discretize_times,
)


class SpikeEncodingDataset(Dataset):
    def __init__(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        image_shape: tuple[int, int] | None = None,
        cache_path: str | None = None,
    ):
        self.outputs = outputs.long()
        self.inputs = inputs.float()

        if image_shape is not None:
            self.inputs = F.interpolate(
                self.inputs.unsqueeze(1),
                size=image_shape,
                mode="nearest",
            ).squeeze(1)

        if cache_path and os.path.exists(cache_path):
            self.all_times = torch.load(cache_path, weights_only=True)
        else:
            dog = apply_difference_of_gaussians_filter_batch(self.inputs)
            self.all_times = discretize_times(apply_latency_encoding(dog))
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(self.all_times, cache_path)

    @property
    def image_shape(self) -> tuple[int, int]:
        """Return (H, W) of the stored images."""
        return (self.inputs.shape[1], self.inputs.shape[2])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.all_times[idx], self.outputs[idx]
