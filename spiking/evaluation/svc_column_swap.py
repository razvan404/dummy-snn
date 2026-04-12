"""LinearSVC with column-swap support via refit.

Unlike Ridge, there is no analytical shortcut for SVM column swaps —
each swap triggers a full refit.  GPU-accelerated via cuML when available.
"""

import numpy as np
from sklearn.svm import LinearSVC as SklearnLinearSVC

from spiking.evaluation.column_swap_classifier import ColumnSwapClassifier

try:
    from cuml.svm import LinearSVC as CumlLinearSVC

    _HAS_CUML = True
except ImportError:
    _HAS_CUML = False

# Parameters supported by cuml.svm.LinearSVC
_CUML_SVC_PARAMS = {"C", "max_iter", "tol", "verbose", "penalty", "loss"}


class SVCColumnSwap(ColumnSwapClassifier):
    """LinearSVC wrapper with column-swap evaluation and optional GPU.

    Column swaps are implemented by refitting the SVM on modified training
    data.  When *use_gpu=True* and cuML is installed, fitting and prediction
    run on the GPU via ``cuml.svm.LinearSVC``.

    Default hyperparameters match the project convention::

        LinearSVC(dual=False, tol=1e-3, max_iter=10_000)

    :param use_gpu: If True, use cuML for GPU-accelerated SVM.
    :param svc_kwargs: Extra keyword arguments forwarded to the underlying
        LinearSVC constructor.
    """

    def __init__(self, use_gpu: bool = False, **svc_kwargs: object):
        self._gpu = use_gpu and _HAS_CUML
        defaults: dict = {"dual": False, "tol": 1e-3, "max_iter": 10_000}
        defaults.update(svc_kwargs)
        self._svc_kwargs = defaults

    def _make_clf(self):
        if self._gpu:
            kw = {k: v for k, v in self._svc_kwargs.items() if k in _CUML_SVC_PARAMS}
            return CumlLinearSVC(**kw)
        return SklearnLinearSVC(**self._svc_kwargs)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVCColumnSwap":
        """Fit the SVM and store training data for future column swaps."""
        self._X = X.astype(np.float64, copy=True)
        self._y = y.copy()
        self._clf = self._make_clf()
        self._clf.fit(self._X, self._y)
        return self

    def predict(self, X_val: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.asarray(self._clf.predict(X_val))

    def predict_swapped(
        self,
        col_indices: list[int] | np.ndarray,
        new_train_cols: np.ndarray,
        X_val_mod: np.ndarray,
    ) -> np.ndarray:
        """Predict after replacing training columns (full refit)."""
        col_indices = np.asarray(col_indices)
        X_mod = self._X.copy()
        X_mod[:, col_indices] = new_train_cols.astype(np.float64)
        clf = self._make_clf()
        clf.fit(X_mod, self._y)
        return np.asarray(clf.predict(X_val_mod))

    def apply_swap(
        self,
        col_indices: list[int] | np.ndarray,
        new_train_cols: np.ndarray,
    ) -> None:
        """Permanently replace training columns and refit."""
        col_indices = np.asarray(col_indices)
        self._X[:, col_indices] = new_train_cols.astype(np.float64)
        self._clf = self._make_clf()
        self._clf.fit(self._X, self._y)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights(self) -> np.ndarray:
        """Weight matrix as numpy array, shape (d, K).

        Transposes sklearn's (K, d) convention to match Ridge's (d, K).
        """
        w = np.asarray(self._clf.coef_)  # (K, d) or (1, d) for binary
        return w.T
