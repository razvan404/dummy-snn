"""GPU-accelerated LinearSVC via Newton-IRLS on the L2-loss SVM primal.

Solves the same problem as sklearn's ``LinearSVC(dual=False, loss='squared_hinge')``
but ~2000x faster on GPU by exploiting:
  - TF32 tensor cores (A100/H100) for Hessian formation
  - Cholesky batched solve across all OVR classes simultaneously
  - Active-set convergence detection for early stopping
  - Warm-start from previous solution for column-swap evaluations
"""

import numpy as np
import torch

from spiking.evaluation.column_swap_classifier import ColumnSwapClassifier

_CUDA_AVAILABLE = torch.cuda.is_available()


class TorchLinearSVC(ColumnSwapClassifier):
    """L2-regularized L2-loss linear SVM solved on GPU via Newton-IRLS.

    Keeps training data on GPU after ``fit()`` for efficient column-swap
    evaluations with warm-starting.  Each swap changes only a few columns,
    so the Newton solver converges in 2-3 iterations instead of ~10.

    :param C: Regularization parameter (inverse of regularization strength).
    :param max_iter: Maximum Newton iterations (cold start).
    :param warm_max_iter: Maximum Newton iterations when warm-starting.
    :param active_tol: Stop when active-set changes by <= this many entries.
    :param gnorm_tol: Stop when gradient norm drops below this.
    :param device: Torch device. Defaults to CUDA if available, else CPU.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 30,
        warm_max_iter: int = 5,
        active_tol: int = 10,
        gnorm_tol: float = 1e-2,
        standardize: bool = True,
        device: torch.device | str | None = None,
    ):
        if device is None:
            device = "cuda" if _CUDA_AVAILABLE else "cpu"
        self._device = torch.device(device)
        self._C = C
        self._max_iter = max_iter
        self._warm_max_iter = warm_max_iter
        self._active_tol = active_tol
        self._gnorm_tol = gnorm_tol
        self._standardize = standardize  # per-column min-max to [0, 1] (Falez 2020)
        torch.backends.cuda.matmul.allow_tf32 = True

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    def _solve_l2svm(
        self,
        Xa: torch.Tensor,
        Y: torch.Tensor,
        W_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve L2-reg L2-loss SVM for all classes via Newton-IRLS.

        :param Xa: Augmented feature matrix (n, d+1) with bias column appended.
        :param Y: OVR label matrix (n, K) with +1/-1.
        :param W_init: Optional warm-start weights (d+1, K). If provided,
            uses ``warm_max_iter`` and tighter active-set tolerance.
        :returns: Weight matrix (d+1, K) including bias row.
        """
        da = Xa.shape[1]
        d = da - 1
        K = Y.shape[1]
        C = self._C

        warm = W_init is not None
        W = W_init.clone() if warm else torch.zeros(da, K, device=Xa.device)
        max_iter = self._warm_max_iter if warm else self._max_iter
        active_tol = 0 if warm else self._active_tol

        # Diagonal regularization: 1 on weights, small jitter on bias
        I_diag = torch.eye(da, device=Xa.device)
        I_diag[d, d] = 1e-4
        I_diag += 1e-4 * torch.eye(da, device=Xa.device)

        prev_active = None

        for _ in range(max_iter):
            scores = Xa @ W
            margin = Y * scores
            active = margin < 1.0
            active_float = active.float()

            if prev_active is not None:
                if (active != prev_active).sum().item() <= active_tol:
                    break
            prev_active = active.clone()

            residual = active_float * (margin - 1.0) * Y
            G = W.clone()
            G[d, :] = 0
            G += 2.0 * C * (Xa.T @ residual)

            if G.norm().item() < self._gnorm_tol:
                break

            H_batch = I_diag.unsqueeze(0).expand(K, -1, -1).clone()
            for k in range(K):
                XaW = Xa * active_float[:, k].unsqueeze(1)
                H_batch[k] += 2.0 * C * (XaW.T @ Xa)

            rhs = -G.T.unsqueeze(2)
            try:
                L = torch.linalg.cholesky(H_batch)
                delta = torch.cholesky_solve(rhs, L).squeeze(2).T
            except torch._C._LinAlgError:
                try:
                    delta = torch.linalg.solve(H_batch, rhs).squeeze(2).T
                except torch._C._LinAlgError:
                    # Hessian is singular (flat direction). Add stronger
                    # diagonal regularization and retry; as last resort,
                    # skip this Newton step.
                    H_batch = H_batch + 1e-2 * torch.eye(
                        da, device=Xa.device
                    ).unsqueeze(0)
                    try:
                        L = torch.linalg.cholesky(H_batch)
                        delta = torch.cholesky_solve(rhs, L).squeeze(2).T
                    except torch._C._LinAlgError:
                        break  # give up, return current W

            step = 1.0
            old_obj = (
                0.5 * (W[:d] ** 2).sum()
                + C * (active_float * (1.0 - margin)).pow(2).sum()
            )
            gTd = (G * delta).sum()
            for _ in range(20):
                W_new = W + step * delta
                margin_new = Y * (Xa @ W_new)
                new_obj = (
                    0.5 * (W_new[:d] ** 2).sum()
                    + C * torch.clamp(1.0 - margin_new, min=0).pow(2).sum()
                )
                if new_obj <= old_obj + 1e-4 * step * gTd:
                    break
                step *= 0.5

            W = W + step * delta

        return W

    def _build_gpu_state(self, X_t: torch.Tensor, y_t: torch.Tensor) -> None:
        """Build cached GPU tensors from training data."""
        n = X_t.shape[0]
        K = int(y_t.max().item()) + 1
        self._X_t = X_t
        self._y_t = y_t
        self._K = K
        # Augmented matrix with bias column
        self._Xa = torch.cat([X_t, torch.ones(n, 1, device=X_t.device)], dim=1)
        # OVR labels
        self._Y = -torch.ones(n, K, device=X_t.device)
        self._Y[torch.arange(n, device=X_t.device), y_t] = 1.0

    # ------------------------------------------------------------------
    # Feature scaling (per-column min-max → [0, 1], Falez 2020)
    # ------------------------------------------------------------------

    def _fit_scaler(self, X: np.ndarray) -> None:
        """Learn per-column min/max from X and cache on CPU + GPU."""
        if not self._standardize:
            self._feat_min = None
            return
        self._feat_min = X.min(axis=0).astype(np.float32)
        self._feat_max = X.max(axis=0).astype(np.float32)
        rng = self._feat_max - self._feat_min
        self._feat_const = rng == 0
        self._feat_range = np.where(self._feat_const, 1.0, rng).astype(np.float32)
        # GPU copies for fast per-swap scaling
        self._feat_min_t = torch.from_numpy(self._feat_min).to(self._device)
        self._feat_range_t = torch.from_numpy(self._feat_range).to(self._device)
        self._feat_const_t = torch.from_numpy(self._feat_const).to(self._device)

    def _scale_np(self, X: np.ndarray) -> np.ndarray:
        """Apply learned min-max scaling to a numpy feature matrix."""
        if not self._standardize or self._feat_min is None:
            return X.astype(np.float32, copy=True)
        out = (X.astype(np.float32) - self._feat_min) / self._feat_range
        out[:, self._feat_const] = 0.0
        return out

    def _scale_cols_np(
        self, cols: np.ndarray, col_indices: np.ndarray
    ) -> np.ndarray:
        """Scale a subset of columns (n, k) using stored min/max for those cols."""
        if not self._standardize or self._feat_min is None:
            return cols.astype(np.float32, copy=True)
        col_indices = np.asarray(col_indices)
        mins = self._feat_min[col_indices]
        rng = self._feat_range[col_indices]
        const = self._feat_const[col_indices]
        out = (cols.astype(np.float32) - mins) / rng
        out[:, const] = 0.0
        return out

    def _scale_cols_gpu(
        self, cols: torch.Tensor, col_indices: torch.Tensor
    ) -> torch.Tensor:
        """Scale (n, k) GPU columns using cached min/max at col_indices."""
        if not self._standardize or self._feat_min is None:
            return cols
        mins = self._feat_min_t[col_indices]  # (k,)
        rng = self._feat_range_t[col_indices]
        const = self._feat_const_t[col_indices]
        out = (cols - mins) / rng
        if const.any():
            out = torch.where(const.expand_as(out), torch.zeros_like(out), out)
        return out

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchLinearSVC":
        """Fit L2-loss linear SVM on GPU. Keeps data on device for swaps.

        If ``standardize=True`` (default), learns per-column min/max from X
        and rescales features to [0, 1] before fitting — matches Falez 2020
        csnn-simulator's ``FeatureScaling`` preprocessing, which is required
        for the primal Newton-IRLS solver to converge reliably on
        unscaled spike-time pool features.
        """
        X_f32 = X.astype(np.float32, copy=True)
        self._y = y.copy()
        self._fit_scaler(X_f32)
        self._X = self._scale_np(X_f32)

        X_t = torch.from_numpy(self._X).to(self._device)
        y_t = torch.from_numpy(self._y).long().to(self._device)
        self._build_gpu_state(X_t, y_t)

        self._Wa = self._solve_l2svm(self._Xa, self._Y)
        d = X_t.shape[1]
        self._W = self._Wa[:d, :]
        self._b = self._Wa[d, :]
        return self

    def predict(self, X_val: np.ndarray) -> np.ndarray:
        """Predict class labels (applies learned scaling)."""
        X_scaled = self._scale_np(np.asarray(X_val))
        X_t = torch.from_numpy(X_scaled).to(self._device)
        with torch.no_grad():
            preds = (X_t @ self._W + self._b).argmax(dim=1)
        return preds.cpu().numpy()

    def predict_swapped(
        self,
        col_indices: list[int] | np.ndarray,
        new_train_cols: np.ndarray,
        X_val_mod: np.ndarray,
    ) -> np.ndarray:
        """Predict after replacing training columns (warm-start refit).

        Inputs are UNSCALED; learned min-max is applied internally.
        """
        col_indices = np.asarray(col_indices)
        col_idx = torch.as_tensor(col_indices, dtype=torch.long, device=self._device)
        new_scaled = self._scale_cols_np(new_train_cols, col_indices)
        new_cols_t = torch.from_numpy(new_scaled).to(self._device)

        old_cols = self._X_t[:, col_idx].clone()
        self._X_t[:, col_idx] = new_cols_t
        self._Xa[:, col_idx] = new_cols_t

        Wa_new = self._solve_l2svm(self._Xa, self._Y, W_init=self._Wa)

        self._X_t[:, col_idx] = old_cols
        self._Xa[:, col_idx] = old_cols

        d = self._X_t.shape[1]
        W_new, b_new = Wa_new[:d, :], Wa_new[d, :]

        X_val_scaled = self._scale_np(np.asarray(X_val_mod))
        X_val_t = torch.from_numpy(X_val_scaled).to(self._device)
        with torch.no_grad():
            preds = (X_val_t @ W_new + b_new).argmax(dim=1)
        return preds.cpu().numpy()

    def eval_swapped_train_acc(
        self,
        col_indices: torch.Tensor,
        new_train_cols: torch.Tensor,
        y_train: torch.Tensor,
    ) -> float:
        """Evaluate train accuracy after a column swap (warm-start, GPU-only).

        ``new_train_cols`` is an UNSCALED (n, k) GPU tensor; scaling is
        applied internally using the stored min/max for those columns.
        """
        new_scaled = self._scale_cols_gpu(new_train_cols, col_indices)
        old_cols = self._X_t[:, col_indices].clone()
        self._X_t[:, col_indices] = new_scaled
        self._Xa[:, col_indices] = new_scaled

        Wa_new = self._solve_l2svm(self._Xa, self._Y, W_init=self._Wa)

        with torch.no_grad():
            preds = (self._Xa @ Wa_new).argmax(dim=1)
            acc = (preds == y_train).float().mean().item()

        self._X_t[:, col_indices] = old_cols
        self._Xa[:, col_indices] = old_cols
        return acc

    # ------------------------------------------------------------------
    # Fast greedy evaluation via precomputed Hessian + single Newton step
    # ------------------------------------------------------------------

    def precompute_hessian(self, col_indices: torch.Tensor) -> None:
        """Precompute baseline Hessian and active set for a neuron sweep.

        Call once before evaluating multiple candidate levels for the same
        neuron.  Subsequent calls to ``eval_swapped_fast`` use the cached
        Hessian with a cheap rank-k correction instead of rebuilding it.
        """
        d = self._X_t.shape[1]
        da = d + 1
        K = self._K
        C = self._C
        Xa = self._Xa

        scores = Xa @ self._Wa
        margin = self._Y * scores
        self._hc_D = (margin < 1.0).float()
        self._hc_scores = scores
        self._hc_col_idx = col_indices
        self._hc_old_cols = Xa[:, col_indices].clone()

        I_diag = torch.eye(da, device=Xa.device)
        I_diag[d, d] = 1e-4
        I_diag += 1e-4 * torch.eye(da, device=Xa.device)

        self._hc_H = I_diag.unsqueeze(0).expand(K, -1, -1).clone()
        for k in range(K):
            XaD = Xa * self._hc_D[:, k].unsqueeze(1)
            self._hc_H[k] += 2.0 * C * (XaD.T @ Xa)

    def eval_swapped_fresh_active(
        self,
        col_indices: torch.Tensor,
        new_train_cols: torch.Tensor,
        y_train: torch.Tensor,
    ) -> float:
        """Single Newton step with freshly-computed active set.

        Unlike ``eval_swapped_fast``, recomputes the active set D_k from
        the swapped features (with baseline W) rather than reusing the
        baseline D.  Correct for all neurons (including top-importance).

        Timing: ~50ms per eval on CIFAR-10 (vs 250ms for full warm-start,
        vs 30ms for stale-D fast path).  5x speedup over full warm-start.
        """
        n, d = self._X_t.shape
        da = d + 1
        K = self._K
        C = self._C
        Xa = self._Xa
        Wa = self._Wa

        # Swap columns in Xa (in-place)
        old_cols = Xa[:, col_indices].clone()
        Xa[:, col_indices] = new_train_cols

        # Compute active set D at swapped state (using baseline W)
        scores = Xa @ Wa
        margin = self._Y * scores
        active = (margin < 1.0).float()

        # Build Hessian with fresh D
        I_diag = torch.eye(da, device=Xa.device)
        I_diag[d, d] = 1e-4
        I_diag += 1e-4 * torch.eye(da, device=Xa.device)
        H = I_diag.unsqueeze(0).expand(K, -1, -1).clone()
        for k in range(K):
            XaD = Xa * active[:, k].unsqueeze(1)
            H[k] += 2.0 * C * (XaD.T @ Xa)

        # Gradient at swapped state
        residual = active * (margin - 1.0) * self._Y
        G = Wa.clone()
        G[d, :] = 0
        G += 2.0 * C * (Xa.T @ residual)

        # Newton step
        rhs = -G.T.unsqueeze(2)
        try:
            L = torch.linalg.cholesky(H)
            delta_W = torch.cholesky_solve(rhs, L).squeeze(2).T
        except torch._C._LinAlgError:
            delta_W = torch.linalg.solve(H, rhs).squeeze(2).T

        Wa_new = Wa + delta_W

        with torch.no_grad():
            preds = (Xa @ Wa_new).argmax(dim=1)
            acc = (preds == y_train).float().mean().item()

        # Restore columns
        Xa[:, col_indices] = old_cols
        return acc

    def eval_swapped_fast(
        self,
        new_train_cols: torch.Tensor,
        y_train: torch.Tensor,
    ) -> float:
        """Single-Newton-step train accuracy with precomputed Hessian.

        WARNING: uses the baseline active set, which is inaccurate for
        high-importance neurons where perturbations shift many samples
        across the margin.  Prefer ``eval_swapped_fresh_active``.
        """
        col_idx = self._hc_col_idx
        old_cols = self._hc_old_cols
        D = self._hc_D
        Xa = self._Xa
        Wa = self._Wa
        C = self._C
        K = self._K
        d = self._X_t.shape[1]

        delta = new_train_cols - old_cols  # (n, k)

        # Correct Hessian rows/cols at col_idx
        H_new = self._hc_H.clone()
        for k in range(K):
            Dk = D[:, k]
            old_w = old_cols * Dk.unsqueeze(1)
            new_w = new_train_cols * Dk.unsqueeze(1)

            new_row = new_w.T @ Xa
            new_row[:, col_idx] = new_w.T @ new_train_cols
            old_row = old_w.T @ Xa
            old_row[:, col_idx] = old_w.T @ old_cols

            delta_row = 2.0 * C * (new_row - old_row)
            H_new[k, col_idx, :] += delta_row
            H_new[k, :, col_idx] += delta_row.T
            delta_block = 2.0 * C * (new_w.T @ new_train_cols - old_w.T @ old_cols)
            H_new[k][col_idx.unsqueeze(1), col_idx.unsqueeze(0)] -= delta_block

        # Gradient at swapped state (reusing baseline active set)
        scores_new = self._hc_scores + delta @ Wa[col_idx]
        margin_new = self._Y * scores_new
        residual_new = D * (margin_new - 1.0) * self._Y

        G_new = Wa.clone()
        G_new[d, :] = 0
        XtR = Xa.T @ residual_new
        XtR[col_idx] = new_train_cols.T @ residual_new
        G_new += 2.0 * C * XtR

        # Solve single Newton step
        rhs = -G_new.T.unsqueeze(2)
        try:
            L = torch.linalg.cholesky(H_new)
            delta_W = torch.cholesky_solve(rhs, L).squeeze(2).T
        except torch._C._LinAlgError:
            delta_W = torch.linalg.solve(H_new, rhs).squeeze(2).T

        Wa_new = Wa + delta_W

        # Predict on train (without modifying Xa)
        with torch.no_grad():
            pred_scores = Xa @ Wa_new + delta @ Wa_new[col_idx]
            preds = pred_scores.argmax(dim=1)
            acc = (preds == y_train).float().mean().item()
        return acc

    # ------------------------------------------------------------------
    # Permanent state mutation
    # ------------------------------------------------------------------

    def apply_swap(
        self,
        col_indices: list[int] | np.ndarray,
        new_train_cols: np.ndarray,
    ) -> None:
        """Permanently replace training columns and refit (warm-start).

        ``new_train_cols`` is UNSCALED (raw pool-spike values); scaling is
        applied internally using the stored min/max for those columns.

        Warm-starts from the current ``self._Wa`` for speed (~300 ms per
        call). Warm-start drift across many commits in one pass is cleaned
        up by a cold refit at pass boundaries (see ``main()`` in
        ``applications/cached_greedy_optimization/optimize.py``).
        """
        col_indices = np.asarray(col_indices)
        new_scaled = self._scale_cols_np(new_train_cols, col_indices)
        self._X[:, col_indices] = new_scaled

        col_idx = torch.as_tensor(col_indices, dtype=torch.long, device=self._device)
        new_cols_t = torch.from_numpy(new_scaled).to(self._device)
        self._X_t[:, col_idx] = new_cols_t
        self._Xa[:, col_idx] = new_cols_t

        self._Wa = self._solve_l2svm(self._Xa, self._Y, W_init=self._Wa)
        d = self._X_t.shape[1]
        self._W = self._Wa[:d, :]
        self._b = self._Wa[d, :]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights(self) -> np.ndarray:
        """Weight matrix as numpy array, shape (d, K)."""
        return self._W.cpu().numpy()
