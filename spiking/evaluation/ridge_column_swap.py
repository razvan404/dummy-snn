import numpy as np
from sklearn.preprocessing import LabelBinarizer


class RidgeColumnSwap:
    """Ridge classifier optimized for evaluating many column-swap perturbations.

    Fits the baseline classifier once in O(d³ + nd²), then uses the Woodbury
    identity to efficiently update predictions when a subset of feature columns
    is replaced with new values. Each column-swap prediction costs O(d²k + md)
    instead of O(d³ + nd²) for a full refit, where k = number of swapped columns.

    This is useful for perturbation analysis where you repeatedly swap one
    neuron/filter's features while keeping the rest fixed.

    Args:
        alpha: Regularization strength (same as sklearn RidgeClassifier).
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeColumnSwap":
        """Fit baseline classifier and precompute inverse.

        Args:
            X: (n, d) training features.
            y: (n,) integer class labels.

        Returns:
            self
        """
        n, d = X.shape
        self._binarizer = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self._binarizer.fit_transform(y).astype(np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.classes_ = self._binarizer.classes_

        # Store centered data for Woodbury updates
        self._X = X.astype(np.float64)
        self._X_mean = X.mean(axis=0).astype(np.float64)
        self._Y_mean = Y.mean(axis=0).astype(np.float64)

        X_c = self._X - self._X_mean
        Y_c = Y - self._Y_mean

        # Precompute inverse: A = X_c^T X_c + αI
        self._X_c = X_c
        self._Y_c = Y_c
        XtX = X_c.T @ X_c
        A = XtX + self.alpha * np.eye(d)
        self._A_inv = np.linalg.inv(A)

        # Precompute X_c^T Y_c for weight computation
        self._XtY_c = X_c.T @ Y_c  # (d, K)

        # Baseline weights: w = A_inv @ X_c^T Y_c, shape (d, K)
        self._w = self._A_inv @ self._XtY_c
        self._intercept = self._Y_mean - self._X_mean @ self._w  # (K,)

        return self

    def predict(self, X_val: np.ndarray) -> np.ndarray:
        """Predict class labels using baseline weights.

        Args:
            X_val: (m, d) validation features.

        Returns:
            (m,) predicted class labels.
        """
        scores = X_val @ self._w + self._intercept  # (m, K)
        return self._decode(scores)

    def predict_swapped(
        self,
        col_indices: list[int] | np.ndarray,
        new_train_cols: np.ndarray,
        X_val_mod: np.ndarray,
    ) -> np.ndarray:
        """Predict after replacing columns in training data using Woodbury update.

        Efficiently computes the Ridge solution that would result from replacing
        columns `col_indices` of the training matrix with `new_train_cols`, without
        refitting from scratch.

        Cost: O(d²k + mk) where k = len(col_indices), m = X_val_mod.shape[0].

        Args:
            col_indices: Indices of columns to replace.
            new_train_cols: (n, k) new values for training data columns.
            X_val_mod: (m, d) validation features with the same columns already replaced.

        Returns:
            (m,) predicted class labels.
        """
        col_indices = np.asarray(col_indices)
        k = len(col_indices)
        d = self._X_c.shape[1]

        # Center new columns
        new_mean = new_train_cols.mean(axis=0).astype(np.float64)
        new_cols_c = new_train_cols.astype(np.float64) - new_mean

        # Column differences (centered space)
        old_cols_c = self._X_c[:, col_indices]  # (n, k)
        D_c = new_cols_c - old_cols_c  # (n, k)

        # Indicator matrix E: (d, k), E[col_indices[i], i] = 1
        E = np.zeros((d, k), dtype=np.float64)
        E[col_indices, np.arange(k)] = 1.0

        # S = X_c^T D_c, shape (d, k)
        S = self._X_c.T @ D_c

        # Woodbury: A_new_inv = A_inv - A_inv U (C^{-1} + U^T A_inv U)^{-1} U^T A_inv
        # where U = [E, S], C = [[D_c^T D_c, I_k], [I_k, 0]]
        # and C^{-1} = [[0, I_k], [I_k, -D_c^T D_c]]
        DtD = D_c.T @ D_c  # (k, k)

        U = np.column_stack([E, S])  # (d, 2k)
        C_inv = np.block([[np.zeros((k, k)), np.eye(k)], [np.eye(k), -DtD]])  # (2k, 2k)

        A_inv_U = self._A_inv @ U  # (d, 2k)
        inner = C_inv + U.T @ A_inv_U  # (2k, 2k)
        inner_inv = np.linalg.inv(inner)  # O(k³)
        A_new_inv = self._A_inv - A_inv_U @ inner_inv @ A_inv_U.T  # (d, d)

        # Update X_c^T Y_c: only rows at col_indices change
        XtY_c_new = self._XtY_c.copy()
        XtY_c_new[col_indices] = new_cols_c.T @ self._Y_c

        # New weights and intercept
        w_new = A_new_inv @ XtY_c_new  # (d, K)
        X_mean_new = self._X_mean.copy()
        X_mean_new[col_indices] = new_mean
        intercept_new = self._Y_mean - X_mean_new @ w_new  # (K,)

        scores = X_val_mod @ w_new + intercept_new  # (m, K)
        return self._decode(scores)

    def apply_swap(
        self,
        col_indices: list[int] | np.ndarray,
        new_train_cols: np.ndarray,
    ) -> None:
        """Permanently apply a column swap, updating all internal state.

        After this call, the classifier behaves as if it was fitted on training
        data with the specified columns replaced. Subsequent predict_swapped calls
        will use this updated state as the new baseline.

        This enables sequential/greedy optimization: optimize neuron 0, apply it,
        then optimize neuron 1 with neuron 0's improvement already baked in.

        Cost: O(d²k) — same as predict_swapped, no O(d³) refit needed.

        Args:
            col_indices: Indices of columns to replace permanently.
            new_train_cols: (n, k) new column values for training data.
        """
        col_indices = np.asarray(col_indices)
        k = len(col_indices)
        d = self._X_c.shape[1]

        # Center new columns
        new_mean = new_train_cols.mean(axis=0).astype(np.float64)
        new_cols_c = new_train_cols.astype(np.float64) - new_mean

        # Column differences in centered space
        old_cols_c = self._X_c[:, col_indices]
        D_c = new_cols_c - old_cols_c

        # Indicator matrix E
        E = np.zeros((d, k), dtype=np.float64)
        E[col_indices, np.arange(k)] = 1.0

        S = self._X_c.T @ D_c
        DtD = D_c.T @ D_c

        # Woodbury update of A_inv
        U = np.column_stack([E, S])
        C_inv = np.block([[np.zeros((k, k)), np.eye(k)], [np.eye(k), -DtD]])
        A_inv_U = self._A_inv @ U
        inner = C_inv + U.T @ A_inv_U
        inner_inv = np.linalg.inv(inner)
        self._A_inv = self._A_inv - A_inv_U @ inner_inv @ A_inv_U.T

        # Update centered training data and mean
        self._X[:, col_indices] = new_train_cols.astype(np.float64)
        self._X_mean[col_indices] = new_mean
        self._X_c[:, col_indices] = new_cols_c

        # Update X_c^T Y_c
        self._XtY_c[col_indices] = new_cols_c.T @ self._Y_c

        # Recompute weights and intercept
        self._w = self._A_inv @ self._XtY_c
        self._intercept = self._Y_mean - self._X_mean @ self._w

    def _decode(self, scores: np.ndarray) -> np.ndarray:
        """Convert score matrix to class labels."""
        if scores.shape[1] == 1:
            # Binary case
            return self.classes_[(scores.ravel() > 0).astype(int)]
        return self.classes_[scores.argmax(axis=1)]
