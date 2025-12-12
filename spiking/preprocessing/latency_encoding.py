import torch


def apply_latency_encoding(input_data: torch.Tensor) -> torch.Tensor:
    times = torch.clamp(1.0 - input_data, min=0.0)
    times[times == 1.0] = float("inf")
    return times
