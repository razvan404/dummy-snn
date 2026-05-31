import torch


def choose_random_winner(spiking_times: torch.Tensor) -> int | None:
    """Return earliest-spiking neuron index, random-tie-break; None if no spike."""
    min_idx = spiking_times.argmin().item()
    min_time = spiking_times[min_idx]
    if torch.isinf(min_time):
        return None
    matches = (spiking_times == min_time)
    num_matches = matches.sum().item()
    if num_matches <= 1:
        return min_idx
    min_indices = torch.nonzero(matches, as_tuple=False).squeeze()
    if min_indices.ndim == 0:
        return min_indices.item()
    return min_indices[torch.randint(len(min_indices), size=(1,))].item()
