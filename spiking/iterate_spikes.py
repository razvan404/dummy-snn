import torch


def iterate_spikes(times: torch.Tensor, shape=None):
    """Yields (incoming_spikes, current_time, delta_time) for each unique spike time.

    Precomputes index grouping so each frame costs O(K_i) instead of O(N).

    :param times: tensor of spike times (e.g. shape (2, H, W)). Non-spiking entries are inf.
    :param shape: unused, kept for call-site compatibility.
    """
    flat = times.flatten()
    valid_indices = torch.nonzero(torch.isfinite(flat), as_tuple=True)[0]
    if len(valid_indices) == 0:
        return

    valid_times = flat[valid_indices]
    unique_times, inverse = torch.unique(valid_times, return_inverse=True, sorted=True)

    # Group valid indices by their time bucket
    sorted_order = torch.argsort(inverse)
    sorted_indices = valid_indices[sorted_order]
    counts = torch.bincount(inverse, minlength=len(unique_times))
    offsets = torch.zeros(len(unique_times) + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts, dim=0)

    prev_time = 0.0
    for i in range(len(unique_times)):
        frame = torch.zeros_like(flat)
        frame[sorted_indices[offsets[i]:offsets[i + 1]]] = 1.0
        current_time = unique_times[i].item()
        yield frame.view(times.shape), current_time, current_time - prev_time
        prev_time = current_time
