import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from spiking.preprocessing import (
    apply_difference_of_gaussians_filter,
    apply_latency_encoding,
)
from spiking import convert_to_spikes


class SpikeEncodingDataset(Dataset):
    def __init__(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        image_shape: tuple[int, int] | None = None,
    ):
        self.outputs = outputs.long()
        self.inputs = inputs.float()

        if image_shape is not None:
            self.inputs = F.interpolate(
                self.inputs.unsqueeze(1),
                size=image_shape,
                mode="nearest",
            ).squeeze(1)

    @property
    def image_shape(self) -> tuple[int, int]:
        """Return (H, W) of the stored images."""
        return (self.inputs.shape[1], self.inputs.shape[2])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        image = self.inputs[idx]
        label = self.outputs[idx]

        dog_image = apply_difference_of_gaussians_filter(image)
        times = apply_latency_encoding(dog_image)
        spikes = convert_to_spikes(times)

        return spikes, label, times
