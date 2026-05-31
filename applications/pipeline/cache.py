from __future__ import annotations

import pickle
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch


def feature_cache_filename(step_size: float, max_drift: float) -> str:
    return f"feature_cache_step{step_size}_drift{max_drift}.pt"


@dataclass
class FeatureCache:
    """The (F, L, N, P) per-(neuron, offset) feature tensor and its labels/metadata.

    Single owner of the on-disk schema: writers (`feature_cache` builder) and readers
    (`alternating_minimization`, `greedy`) share these field names instead of
    re-spelling dict-key string literals. Field names equal the legacy dict keys, so
    existing `.pt` caches load unchanged.
    """

    train_cache: np.ndarray  # (F, L, N_train, P)
    test_cache: np.ndarray  # (F, L, N_test, P)
    y_train: np.ndarray
    y_test: np.ndarray
    original_thresholds: np.ndarray
    perturbation_fractions: list[float]
    step_size: float
    max_drift: float
    pool_size: int
    t_target: float

    def save(self, path: str | Path) -> None:
        torch.save(
            {f.name: getattr(self, f.name) for f in fields(self)},
            path,
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureCache":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**{f.name: d[f.name] for f in fields(cls)})

    @property
    def zero_index(self) -> int:
        f = np.asarray(self.perturbation_fractions)
        return int(np.argmin(np.abs(f)))
