from abc import ABC, abstractmethod


class ThresholdInitialization(ABC):
    @abstractmethod
    def initialize(self, threshold: float): ...
