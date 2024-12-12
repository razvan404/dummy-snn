from abc import ABC

import numpy as np


class CompetitionMechanism(ABC):
    def neurons_to_learn(self, spike_times: np.ndarray): ...
