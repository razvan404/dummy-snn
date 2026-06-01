import torch
from torchvision.transforms import v2


class LatencyEncoding(v2.Transform):
    def transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        times = torch.clamp(1.0 - inpt, min=0.0)
        times[times == 1.0] = float("inf")  # zero intensity never spikes
        return times
