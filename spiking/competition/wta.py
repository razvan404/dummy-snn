import numpy as np

from .mechansim import CompetitionMechanism


class WinnerTakesAll(CompetitionMechanism):
    def neurons_to_learn(self, spiking_times: np.ndarray):
        min_time = spiking_times.min()
        min_indices = np.where(spiking_times == min_time)[0]
        selected_index = np.random.choice(min_indices)
        return [selected_index]
