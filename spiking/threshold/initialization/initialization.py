from abc import ABC, abstractmethod

import torch


class ThresholdInitialization(ABC):
    @abstractmethod
    def initialize(self, shape: tuple[int] | int = 1) -> torch.Tensor | float:
        pass
