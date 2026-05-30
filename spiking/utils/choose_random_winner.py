import torch


def choose_random_winner(spiking_times: torch.Tensor) -> int | None:
    """Return earliest-spiking neuron index, random-tie-break; None if no spike."""
    min_time = spiking_times.min()
    if torch.isinf(min_time):
        return None
    min_indices = torch.nonzero(spiking_times == min_time, as_tuple=False).squeeze()
    if min_indices.ndim == 0:
        return min_indices.item()
    selected_index = min_indices[torch.randint(len(min_indices), size=(1,))].item()
    return selected_index
