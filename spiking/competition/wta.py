import torch

from .mechansim import CompetitionMechanism
from ..utils import choose_random_winner


class WinnerTakesAll(CompetitionMechanism):
    def neurons_to_learn(self, spiking_times: torch.Tensor) -> torch.Tensor:
        winner_idx = choose_random_winner(spiking_times)
        device = spiking_times.device
        return (
            torch.tensor([winner_idx], dtype=torch.long, device=device)
            if winner_idx is not None
            else torch.tensor([], dtype=torch.long, device=device)
        )
