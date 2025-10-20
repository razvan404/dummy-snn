import torch

from .initialization import ThresholdInitialization
from spiking.registry import registry


@registry.register("threshold.initialization", "normal")
class NormalInitialization(ThresholdInitialization):
    def __init__(
        self, avg_threshold: float, min_threshold: float, std_dev: float = 0.1
    ):
        self.avg_threshold = avg_threshold
        self.min_threshold = min_threshold
        self.std_dev = std_dev

    def initialize(self, shape: tuple[int] | int = 1) -> torch.Tensor | float:
        if isinstance(shape, int):
            shape = (shape,)

        thresholds = torch.normal(mean=self.avg_threshold, std=self.std_dev, size=shape)
        thresholds = torch.clamp(thresholds, min=self.min_threshold)

        if shape == (1,):
            return thresholds.item()
        return thresholds
