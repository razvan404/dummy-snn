import torch
from torchvision.transforms import v2


class DiscretizeTimes(v2.Transform):
    def __init__(self, num_bins: int = 64):
        super().__init__()
        self.num_bins = num_bins

    def transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        result = inpt.clone()
        finite = torch.isfinite(result)  # keep inf (no-spike) entries
        result[finite] = torch.floor(result[finite] * self.num_bins) / self.num_bins
        return result
