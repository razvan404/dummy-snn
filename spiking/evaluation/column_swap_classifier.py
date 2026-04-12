"""Abstract base class for classifiers supporting column-swap evaluation."""

from abc import ABC, abstractmethod

import numpy as np


class ColumnSwapClassifier(ABC):
    """Classifier that supports efficient column-swap perturbation evaluation.

    Column-swap replaces a subset of feature columns in the training data and
    evaluates predictions on modified validation data.  Implementations may use
    analytical shortcuts (e.g. Woodbury for Ridge) or fall back to refitting.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ColumnSwapClassifier":
        """Fit the classifier on training data.

        :param X: (n, d) training features.
        :param y: (n,) integer class labels.
        :returns: self
        """

    @abstractmethod
    def predict(self, X_val: np.ndarray) -> np.ndarray:
        """Predict class labels.

        :param X_val: (m, d) features.
        :returns: (m,) predicted class labels.
        """

    @abstractmethod
    def predict_swapped(
        self,
        col_indices: list[int] | np.ndarray,
        new_train_cols: np.ndarray,
        X_val_mod: np.ndarray,
    ) -> np.ndarray:
        """Predict after virtually replacing training columns.

        Does **not** mutate internal state.

        :param col_indices: Indices of columns to replace.
        :param new_train_cols: (n, k) replacement training column values.
        :param X_val_mod: (m, d) validation features (with same columns replaced).
        :returns: (m,) predicted class labels.
        """

    @abstractmethod
    def apply_swap(
        self,
        col_indices: list[int] | np.ndarray,
        new_train_cols: np.ndarray,
    ) -> None:
        """Permanently apply a column swap, updating internal state.

        Subsequent calls use this updated state as the new baseline.

        :param col_indices: Indices of columns to replace.
        :param new_train_cols: (n, k) replacement training column values.
        """

    @property
    @abstractmethod
    def weights(self) -> np.ndarray:
        """Classifier weight matrix as a numpy array, shape (d, K)."""
