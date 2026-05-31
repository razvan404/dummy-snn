from abc import ABC, abstractmethod

import torch


class CompetitionMechanism(ABC):
    @abstractmethod
    def neurons_to_learn(self, spike_times: torch.Tensor) -> torch.Tensor:
        pass
