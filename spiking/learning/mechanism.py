from abc import ABC, abstractmethod

import numpy as np


class LearningMechanism(ABC):
    @abstractmethod
    def update_weights(
        self,
        weight: np.ndarray,
        pre_spike_times: np.ndarray,
        post_spike_times: np.ndarray,
    ) -> np.ndarray: ...

    @abstractmethod
    def learning_rate_step(self): ...
