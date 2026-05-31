from abc import ABC, abstractmethod

import torch


class LearningMechanism(ABC):
    @abstractmethod
    def update_weights(
        self,
        weights: torch.Tensor,
        pre_spike_times: torch.Tensor,
        post_spike_times: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def learning_rate_step(self):
        pass
