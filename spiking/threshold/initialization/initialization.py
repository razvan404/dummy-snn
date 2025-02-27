from abc import ABC, abstractmethod


class ThresholdInitialization(ABC):
    @abstractmethod
    def initialize(self, threshold: float, shape: tuple[int] | int = 1): ...
