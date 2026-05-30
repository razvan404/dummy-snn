from __future__ import annotations

import torch


_VALID = {"dense", "scatter", "gather"}


def select_backend(
    device: str | torch.device,
    batch_size: int,
    first_spike_only: bool = True,
) -> str:
    """Empirical: CUDA → gather; CPU → gather (B<8) else scatter."""
    dev = torch.device(device).type if not isinstance(device, str) else device
    if dev == "cuda":
        return "gather"
    return "gather" if batch_size < 8 else "scatter"


def is_valid_backend(name: str) -> bool:
    ...
    return name in _VALID
