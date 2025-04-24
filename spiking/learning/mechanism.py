from abc import ABC, abstractmethod

import torch


class LearningMechanism(ABC):
    @abstractmethod
    def update_weights(
        self,
        pre_spike_times: torch.Tensor,
        weight: torch.Tensor,
        post_spike_times: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def learning_rate_step(self):
        pass
