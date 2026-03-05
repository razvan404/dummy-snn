import torch

from .initialization import ThresholdInitialization


class ConstantInitialization(ThresholdInitialization):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def initialize(self, shape: tuple[int] | int = 1) -> torch.Tensor:
        if isinstance(shape, int):
            shape = (shape,)
        return torch.full(shape, self.threshold)
