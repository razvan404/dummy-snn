import torch

from .initialization import ThresholdInitialization
from spiking.registry import registry


@registry.register("threshold.initialization", "constant")
class ConstantInitialization(ThresholdInitialization):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def initialize(self, shape: tuple[int] | int = 1) -> torch.Tensor | float:
        if shape == 1:
            return self.threshold
        return torch.ones(shape) * self.threshold
