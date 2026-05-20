"""LinearSVC with column-swap support — thin facade over ``TorchLinearSVC``.

Kept as a stable API surface for callers that predate ``TorchLinearSVC``.
All work delegates to ``TorchLinearSVC``, which solves the same L2-loss SVM
primal and supports warm-start column swaps (and CUDA when available).
"""

from __future__ import annotations

import numpy as np

from spiking.evaluation.column_swap_classifier import ColumnSwapClassifier
from spiking.evaluation.torch_svc import TorchLinearSVC


class SVCColumnSwap(ColumnSwapClassifier):
    """LinearSVC + column-swap; delegates to :class:`TorchLinearSVC`.

    :param use_gpu: Ignored. ``TorchLinearSVC`` self-detects CUDA.
    :param svc_kwargs: Forwarded to ``TorchLinearSVC`` (``C``, ``max_iter``,
        ``warm_max_iter``, ``standardize``, ``device`` …).
    """

    def __init__(self, use_gpu: bool = False, **svc_kwargs: object) -> None:
        # use_gpu kept for back-compat; TorchLinearSVC picks device itself.
        svc_kwargs.pop("random_state", None)  # sklearn arg; not relevant
        svc_kwargs.pop("dual", None)
        svc_kwargs.pop("tol", None)
        defaults: dict = {"C": 1.0}
        defaults.update(svc_kwargs)
        self._inner = TorchLinearSVC(**defaults)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVCColumnSwap":
        self._inner.fit(X, y)
        return self

    def predict(self, X_val: np.ndarray) -> np.ndarray:
        return self._inner.predict(X_val)

    def predict_swapped(
        self,
        col_indices: list[int] | np.ndarray,
        new_train_cols: np.ndarray,
        X_val_mod: np.ndarray,
    ) -> np.ndarray:
        return self._inner.predict_swapped(col_indices, new_train_cols, X_val_mod)

    def apply_swap(
        self,
        col_indices: list[int] | np.ndarray,
        new_train_cols: np.ndarray,
    ) -> None:
        self._inner.apply_swap(col_indices, new_train_cols)

    @property
    def weights(self) -> np.ndarray:
        return self._inner.weights
