from abc import ABC, abstractmethod

import numpy as np


class ThresholdAdaptation(ABC):
    @abstractmethod
    def update(
        self, current_threshold: np.ndarray, spike_times: np.ndarray, **kwargs
    ) -> np.ndarray: ...

    @abstractmethod
    def learning_rate_step(self): ...
