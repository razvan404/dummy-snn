import torch

from spikinn.evaluation.torch_svc import TorchLinearSVC


class TorchLogisticRegression(TorchLinearSVC):
    _loss_type = "logistic"
    _hess_weight_eps = 0.0  # Hessian-subsampling threshold; 0 = exact

    def __init__(self, *args, hess_weight_eps: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        # only rows with IRLS weight sigma(1-sigma) > eps enter the Hessian; the gradient stays
        # exact so the optimum is unchanged
        self._hess_weight_eps = hess_weight_eps

    def _solve_l2svm(
        self,
        Xa: torch.Tensor,
        Y: torch.Tensor,
        W_init: torch.Tensor | None = None,
        max_iter: int | None = None,
    ) -> torch.Tensor:
        da = Xa.shape[1]
        d = da - 1
        K = Y.shape[1]
        C = self._C

        warm = W_init is not None
        W = W_init.clone() if warm else torch.zeros(da, K, device=Xa.device, dtype=Xa.dtype)
        if max_iter is None:
            max_iter = self._warm_max_iter if warm else self._max_iter

        I_diag = torch.eye(da, device=Xa.device, dtype=Xa.dtype)
        I_diag[d, d] = 1e-4
        I_diag += 1e-4 * torch.eye(da, device=Xa.device, dtype=Xa.dtype)

        for _ in range(max_iter):
            scores = Xa @ W
            margin = Y * scores
            sigma = torch.sigmoid(margin)
            weight = sigma * (1.0 - sigma)

            residual = -(1.0 - sigma) * Y
            G = W.clone()
            G[d, :] = 0
            G += C * (Xa.T @ residual)

            if G.norm().item() < self._gnorm_tol:
                break

            H_batch = I_diag.unsqueeze(0).expand(K, -1, -1).clone()
            eps = self._hess_weight_eps
            for k in range(K):
                if eps > 0.0:
                    # subsampled Newton: drop rows whose curvature weight sigma(1-sigma) <= eps
                    m = weight[:, k] > eps
                    Xm = Xa[m]
                    H_batch[k] += C * (Xm.T @ (Xm * weight[m, k].unsqueeze(1)))
                else:
                    H_batch[k] += C * (Xa.T @ (Xa * weight[:, k].unsqueeze(1)))

            rhs = -G.T.unsqueeze(2)
            try:
                L = torch.linalg.cholesky(H_batch)
                delta = torch.cholesky_solve(rhs, L).squeeze(2).T
            except torch._C._LinAlgError:
                # flag for the float64 refit in fit() (mirrors the SVC solver)
                self._solve_failed = True
                try:
                    delta = torch.linalg.solve(H_batch, rhs).squeeze(2).T
                except torch._C._LinAlgError:
                    H_batch = H_batch + 1e-2 * torch.eye(
                        da, device=Xa.device, dtype=Xa.dtype
                    ).unsqueeze(0)
                    try:
                        L = torch.linalg.cholesky(H_batch)
                        delta = torch.cholesky_solve(rhs, L).squeeze(2).T
                    except torch._C._LinAlgError:
                        break

            # Armijo line search on the logistic primal
            step = 1.0
            old_obj = (
                0.5 * (W[:d] ** 2).sum()
                + C * torch.nn.functional.softplus(-margin).sum()
            )
            gTd = (G * delta).sum()
            for _ in range(20):
                W_new = W + step * delta
                margin_new = Y * (Xa @ W_new)
                new_obj = (
                    0.5 * (W_new[:d] ** 2).sum()
                    + C * torch.nn.functional.softplus(-margin_new).sum()
                )
                if new_obj <= old_obj + 1e-4 * step * gTd:
                    break
                step *= 0.5

            W = W + step * delta

        return W

    def loss_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        score = self._Xa @ self._Wa
        margin = self._Y * score
        sigma = torch.sigmoid(margin)
        coef = -self._C * (1.0 - sigma) * self._Y
        hess_weight = self._C * sigma * (1.0 - sigma)
        return coef, hess_weight
