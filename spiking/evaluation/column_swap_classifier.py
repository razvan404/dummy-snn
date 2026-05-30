from abc import ABC, abstractmethod

import numpy as np


class ColumnSwapClassifier(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ColumnSwapClassifier":
        ...

    @abstractmethod
    def predict(self, X_val: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def predict_swapped(
        self,
        col_indices: list[int] | np.ndarray,
        new_train_cols: np.ndarray,
        X_val_mod: np.ndarray,
    ) -> np.ndarray:
        ...

    @abstractmethod
    def apply_swap(
        self,
        col_indices: list[int] | np.ndarray,
        new_train_cols: np.ndarray,
    ) -> None:
        ...

    @property
    @abstractmethod
    def weights(self) -> np.ndarray:
        ...
