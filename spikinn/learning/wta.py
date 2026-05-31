import torch

from .competition import CompetitionMechanism
from spikinn.utils import choose_random_winner


class WinnerTakesAll(CompetitionMechanism):
    def neurons_to_learn(self, spikinn_times: torch.Tensor) -> torch.Tensor:
        winner_idx = choose_random_winner(spikinn_times)
        device = spikinn_times.device
        return (
            torch.tensor([winner_idx], dtype=torch.long, device=device)
            if winner_idx is not None
            else torch.tensor([], dtype=torch.long, device=device)
        )
