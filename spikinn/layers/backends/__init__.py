from __future__ import annotations

from typing import Callable, Tuple

import torch

from .base import base
from .dense import dense, dense_fp16
from .gather import gather
from .differential_base import differential_base
from .differential_dense import differential_dense

Backend = Callable[..., Tuple[torch.Tensor, torch.Tensor]]


BACKENDS: dict[str, Backend] = {
    "base": base,
    "dense": dense,
    "dense_fp16": dense_fp16,
    "gather": gather,
    "differential_base": differential_base,
    "differential_dense": differential_dense,
}


def get_backend(name: str) -> Backend:
    if name not in BACKENDS:
        raise ValueError(
            f"unknown backend {name!r}; choose from {sorted(BACKENDS)}"
        )
    return BACKENDS[name]


def is_differentiable(name: str) -> bool:
    return name.startswith("differential_")


__all__ = [
    "BACKENDS", "Backend", "get_backend", "is_differentiable",
    "base", "dense", "dense_fp16", "gather",
    "differential_base", "differential_dense",
]
