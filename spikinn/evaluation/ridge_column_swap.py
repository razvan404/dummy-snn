import numpy as np
from sklearn.preprocessing import LabelBinarizer

from spikinn.evaluation.column_swap_classifier import ColumnSwapClassifier

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


def _xp(use_gpu: bool):
    return cp if use_gpu else np


class RidgeColumnSwap(ColumnSwapClassifier):
    def __init__(self, alpha: float = 1.0, use_gpu: bool = False):
        self.alpha = alpha
        self._gpu = use_gpu and _HAS_CUPY
        self._xp = _xp(self._gpu)

    def _to_xp(self, arr: np.ndarray):
        if self._gpu:
            return cp.asarray(arr, dtype=cp.float64)
        return np.asarray(arr, dtype=np.float64)

    def _to_np(self, arr) -> np.ndarray:
        if self._gpu:
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeColumnSwap":
        xp = self._xp

        self._binarizer = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self._binarizer.fit_transform(y).astype(np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.classes_ = self._binarizer.classes_

        X_dev = self._to_xp(X)
        Y_dev = self._to_xp(Y)

        self._X = X_dev
        self._X_mean = X_dev.mean(axis=0)
        self._Y_mean = Y_dev.mean(axis=0)

        X_c = X_dev - self._X_mean
        Y_c = Y_dev - self._Y_mean

        self._X_c = X_c
        self._Y_c = Y_c

        d = X_c.shape[1]
        A = X_c.T @ X_c + self.alpha * xp.eye(d, dtype=xp.float64)
        self._A_inv = xp.linalg.inv(A)
        self._XtY_c = X_c.T @ Y_c
        self._w = self._A_inv @ self._XtY_c
        self._intercept = self._Y_mean - self._X_mean @ self._w

        return self

    def predict(self, X_val: np.ndarray) -> np.ndarray:
        X_dev = self._to_xp(X_val)
        scores = X_dev @ self._w + self._intercept
        return self._decode(self._to_np(scores))

    def predict_swapped(
        self,
        col_indices: list[int] | np.ndarray,
        new_train_cols: np.ndarray,
        X_val_mod: np.ndarray,
    ) -> np.ndarray:
        xp = self._xp
        col_indices = np.asarray(col_indices)
        k = len(col_indices)
        d = self._X_c.shape[1]

        new_dev = self._to_xp(new_train_cols)
        new_mean = new_dev.mean(axis=0)
        new_cols_c = new_dev - new_mean

        D_c = new_cols_c - self._X_c[:, col_indices]

        E = xp.zeros((d, k), dtype=xp.float64)
        E[col_indices, xp.arange(k)] = 1.0

        S = self._X_c.T @ D_c
        DtD = D_c.T @ D_c

        U = xp.column_stack([E, S])
        top = xp.concatenate(
            [xp.zeros((k, k), dtype=xp.float64), xp.eye(k, dtype=xp.float64)], axis=1
        )
        bot = xp.concatenate([xp.eye(k, dtype=xp.float64), -DtD], axis=1)
        C_inv = xp.concatenate([top, bot], axis=0)

        A_inv_U = self._A_inv @ U
        inner_inv = xp.linalg.inv(C_inv + U.T @ A_inv_U)
        A_new_inv = self._A_inv - A_inv_U @ inner_inv @ A_inv_U.T

        XtY_c_new = self._XtY_c.copy()
        XtY_c_new[col_indices] = new_cols_c.T @ self._Y_c

        w_new = A_new_inv @ XtY_c_new
        X_mean_new = self._X_mean.copy()
        X_mean_new[col_indices] = new_mean
        intercept_new = self._Y_mean - X_mean_new @ w_new

        X_val_dev = self._to_xp(X_val_mod)
        scores = X_val_dev @ w_new + intercept_new
        return self._decode(self._to_np(scores))

    def apply_swap(
        self,
        col_indices: list[int] | np.ndarray,
        new_train_cols: np.ndarray,
    ) -> None:
        xp = self._xp
        col_indices = np.asarray(col_indices)
        k = len(col_indices)
        d = self._X_c.shape[1]

        new_dev = self._to_xp(new_train_cols)
        new_mean = new_dev.mean(axis=0)
        new_cols_c = new_dev - new_mean

        D_c = new_cols_c - self._X_c[:, col_indices]

        E = xp.zeros((d, k), dtype=xp.float64)
        E[col_indices, xp.arange(k)] = 1.0

        S = self._X_c.T @ D_c
        DtD = D_c.T @ D_c

        U = xp.column_stack([E, S])
        top = xp.concatenate(
            [xp.zeros((k, k), dtype=xp.float64), xp.eye(k, dtype=xp.float64)], axis=1
        )
        bot = xp.concatenate([xp.eye(k, dtype=xp.float64), -DtD], axis=1)
        C_inv = xp.concatenate([top, bot], axis=0)
        A_inv_U = self._A_inv @ U
        inner_inv = xp.linalg.inv(C_inv + U.T @ A_inv_U)
        self._A_inv = self._A_inv - A_inv_U @ inner_inv @ A_inv_U.T

        self._X[:, col_indices] = new_dev
        self._X_mean[col_indices] = new_mean
        self._X_c[:, col_indices] = new_cols_c
        self._XtY_c[col_indices] = new_cols_c.T @ self._Y_c

        self._w = self._A_inv @ self._XtY_c
        self._intercept = self._Y_mean - self._X_mean @ self._w

    @property
    def weights(self) -> np.ndarray:
        return self._to_np(self._w)

    def _decode(self, scores: np.ndarray) -> np.ndarray:
        if scores.shape[1] == 1:
            return self.classes_[(scores.ravel() > 0).astype(int)]
        return self.classes_[scores.argmax(axis=1)]
