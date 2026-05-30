import torch

from spiking.evaluation.torch_svc import TorchLinearSVC


class TorchLogisticRegression(TorchLinearSVC):
    _loss_type = "logistic"

    def _solve_l2svm(
        self,
        Xa: torch.Tensor,
        Y: torch.Tensor,
        W_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        score = self._Xa @ self._Wa
        margin = self._Y * score
        sigma = torch.sigmoid(margin)
        coef = -self._C * (1.0 - sigma) * self._Y
        hess_weight = self._C * sigma * (1.0 - sigma)
        return coef, hess_weight
