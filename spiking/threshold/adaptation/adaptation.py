from abc import ABC, abstractmethod

import numpy as np


class ThresholdAdaptation(ABC):
    @abstractmethod
    def update(
        self, current_threshold: np.ndarray, spike_times: np.ndarray
    ) -> np.ndarray: ...
