"""GPU-accelerated multinomial OVR logistic regression.

Smooth differentiable analog of TorchLinearSVC — same interface (inherits from
it), but the loss is strictly convex C^∞ everywhere (no active-set kinks).
Used as the optimisation surrogate for threshold tuning when SVM's non-smooth
loss confuses local gradient / curvature analysis. Final classification
accuracy is still reported via TorchLinearSVC.
"""

import torch

from spiking.evaluation.torch_svc import TorchLinearSVC


class TorchLogisticRegression(TorchLinearSVC):
    """L2-regularised OVR logistic regression via Newton-IRLS.

    Loss per (n, k):  log(1 + exp(−Y_nk · score_nk))   (softplus).
    OVR multinomial: each class is an independent binary logistic; argmax
    over class scores at predict time.

    Inherits TorchLinearSVC's standardisation, column-swap, and GPU plumbing.
    Only `_solve_l2svm` is overridden to use the logistic Hessian/gradient
    (sigmoid-based, no margin-active gating).
    """

    _loss_type = "logistic"

    def _solve_l2svm(
        self,
        Xa: torch.Tensor,
        Y: torch.Tensor,
        W_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Newton-IRLS for L2-regularised OVR logistic regression.

        Per-class binary logistic gradient and Hessian:
          residual_nk    = −(1 − σ_nk) · Y_nk         where σ_nk = sigmoid(Y_nk · score_nk)
          hess_weight_nk = σ_nk · (1 − σ_nk)
        Loss: 0.5·||W||² + C·Σ softplus(−Y·score).
        """
        da = Xa.shape[1]
        d = da - 1
        K = Y.shape[1]
        C = self._C

        warm = W_init is not None
        W = W_init.clone() if warm else torch.zeros(da, K, device=Xa.device)
        max_iter = self._warm_max_iter if warm else self._max_iter

        I_diag = torch.eye(da, device=Xa.device)
        I_diag[d, d] = 1e-4
        I_diag += 1e-4 * torch.eye(da, device=Xa.device)

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
            for k in range(K):
                XaW = Xa * weight[:, k].unsqueeze(1)
                H_batch[k] += C * (XaW.T @ Xa)

            rhs = -G.T.unsqueeze(2)
            try:
                L = torch.linalg.cholesky(H_batch)
                delta = torch.cholesky_solve(rhs, L).squeeze(2).T
            except torch._C._LinAlgError:
                try:
                    delta = torch.linalg.solve(H_batch, rhs).squeeze(2).T
                except torch._C._LinAlgError:
                    H_batch = H_batch + 1e-2 * torch.eye(
                        da, device=Xa.device
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
        """Per-(sample, class) (coef, hess_weight) used by EM gradient/curvature.

        For logistic:
          coef_nk        = −C · (1 − σ_nk) · Y_nk
          hess_weight_nk = C · σ_nk · (1 − σ_nk)

        Both are smooth (no boundary kinks). `coef @ Wa.T` gives the gradient
        in scaled feature space.
        """
        score = self._Xa @ self._Wa
        margin = self._Y * score
        sigma = torch.sigmoid(margin)
        coef = -self._C * (1.0 - sigma) * self._Y
        hess_weight = self._C * sigma * (1.0 - sigma)
        return coef, hess_weight
