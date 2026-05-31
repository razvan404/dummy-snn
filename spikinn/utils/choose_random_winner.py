import torch


def choose_random_winner(spikinn_times: torch.Tensor) -> int | None:
    """Return earliest-spikinn neuron index, random-tie-break; None if no spike."""
    min_idx = spikinn_times.argmin().item()
    min_time = spikinn_times[min_idx]
    if torch.isinf(min_time):
        return None
    matches = (spikinn_times == min_time)
    num_matches = matches.sum().item()
    if num_matches <= 1:
        return min_idx
    min_indices = torch.nonzero(matches, as_tuple=False).squeeze()
    if min_indices.ndim == 0:
        return min_indices.item()
    return min_indices[torch.randint(len(min_indices), size=(1,))].item()
