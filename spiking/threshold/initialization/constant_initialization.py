import torch

from .initialization import ThresholdInitialization


class ConstantInitialization(ThresholdInitialization):
    def initialize(
        self, threshold: float, shape: tuple[int] | int = 1
    ) -> torch.Tensor | float:
        if shape == 1:
            return threshold
        return torch.ones(shape) * threshold
