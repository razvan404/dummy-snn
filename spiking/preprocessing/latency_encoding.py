import torch


def apply_latency_encoding(input_data: torch.Tensor) -> torch.Tensor:
    times = torch.clamp(1.0 - input_data, min=0.0)
    times[times == 1.0] = float("inf")
    return times


def discretize_times(times: torch.Tensor, num_bins: int = 64) -> torch.Tensor:
    """Quantize continuous spike times to a fixed number of bins.

    Maps continuous values in [0, 1) to {0/num_bins, 1/num_bins, ..., (num_bins-1)/num_bins}.
    Infinite values (no spike) are preserved.
    """
    result = times.clone()
    finite = torch.isfinite(result)
    result[finite] = torch.floor(result[finite] * num_bins) / num_bins
    return result
