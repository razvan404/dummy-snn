import numpy as np

from .mechansim import CompetitionMechanism


class WinnerTakesAll(CompetitionMechanism):
    def neurons_to_learn(self, spiking_times: np.ndarray):
        return [np.argmin(spiking_times)]
