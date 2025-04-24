from abc import ABC, abstractmethod

import torch


class ThresholdAdaptation(ABC):
    @abstractmethod
    def update(
        self, current_threshold: torch.Tensor, spike_times: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def learning_rate_step(self):
        pass
