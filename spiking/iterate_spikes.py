import torch


def iterate_spikes(times: torch.Tensor):
    """Yields (incoming_spikes, current_time, delta_time) for each unique spike time.

    Precomputes index grouping so each frame costs O(K_i) instead of O(N).

    :param times: tensor of spike times (e.g. shape (2, H, W)). Non-spiking entries are inf.
    """
    flat = times.flatten()
    valid_indices = torch.nonzero(torch.isfinite(flat), as_tuple=True)[0]
    if len(valid_indices) == 0:
        return

    valid_times = flat[valid_indices]
    sorted_times, sort_order = valid_times.sort()
    sorted_indices = valid_indices[sort_order]

    unique_times, counts = torch.unique_consecutive(sorted_times, return_counts=True)
    offsets = torch.zeros(len(unique_times) + 1, dtype=torch.long)
    offsets[1:] = counts.cumsum(dim=0)

    # Pre-allocated frame reused each timestep. Only indices set on the
    # previous iteration are zeroed, keeping cost at O(K) instead of O(N).
    # Callers must consume the yielded view before the next iteration.
    frame = torch.zeros_like(flat)
    prev_time = 0.0
    prev_indices = sorted_indices[offsets[0] : offsets[1]]  # will be zeroed on 2nd iter
    for i in range(len(unique_times)):
        cur_indices = sorted_indices[offsets[i] : offsets[i + 1]]
        if i > 0:
            frame[prev_indices] = 0.0
        frame[cur_indices] = 1.0
        current_time = unique_times[i].item()
        yield frame.view(times.shape), current_time, current_time - prev_time
        prev_time = current_time
        prev_indices = cur_indices
