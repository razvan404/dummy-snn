from abc import ABC, abstractmethod

import torch


class ThresholdInitialization(ABC):
    @abstractmethod
    def initialize(
        self, threshold: float, shape: tuple[int] | int = 1
    ) -> torch.Tensor | float:
        pass
