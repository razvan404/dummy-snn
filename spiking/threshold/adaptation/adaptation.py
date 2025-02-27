from abc import ABC, abstractmethod


class ThresholdAdaptation(ABC):
    @abstractmethod
    def update(self, current_threshold: float, current_time: float) -> float: ...
