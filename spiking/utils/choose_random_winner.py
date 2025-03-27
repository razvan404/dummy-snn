import numpy as np


def choose_random_winner(spiking_times: np.ndarray) -> int | None:
    min_time = spiking_times.min()
    if np.isinf(min_time):
        return None
    min_indices = np.where(spiking_times == min_time)[0]
    selected_index = np.random.choice(min_indices)
    return selected_index