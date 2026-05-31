from __future__ import annotations

import torch


_VALID = {"dense", "gather"}


def select_backend(
    device: str | torch.device,
    batch_size: int,
    first_spike_only: bool = True,
) -> str:
    return "gather"


def is_valid_backend(name: str) -> bool:
    ...
    return name in _VALID
