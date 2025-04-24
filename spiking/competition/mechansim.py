from abc import ABC

import torch


class CompetitionMechanism(ABC):
    def neurons_to_learn(self, spike_times: torch.Tensor) -> torch.Tensor:
        pass
