import torch
from .initialization import ThresholdInitialization


class NormalInitialization(ThresholdInitialization):
    def __init__(self, min_threshold: float, std_dev: float = 0.1):
        self.min_threshold = min_threshold
        self.std_dev = std_dev

    def initialize(
        self, threshold: float, shape: tuple[int] | int = 1
    ) -> torch.Tensor | float:
        if isinstance(shape, int):
            shape = (shape,)

        thresholds = torch.normal(mean=threshold, std=self.std_dev, size=shape)
        thresholds = torch.clamp(thresholds, min=self.min_threshold)

        if shape == (1,):
            return thresholds.item()
        return thresholds
