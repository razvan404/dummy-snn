import numpy as np

from .mechansim import CompetitionMechanism
from ..utils import choose_random_winner


class WinnerTakesAll(CompetitionMechanism):
    def neurons_to_learn(self, spiking_times: np.ndarray) -> np.ndarray:
        winner_idx = choose_random_winner(spiking_times)
        return np.array([winner_idx]) if winner_idx is not None else np.array([])
