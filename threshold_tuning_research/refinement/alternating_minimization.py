from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from threshold_tuning_research.common import set_seed
from threshold_tuning_research.pipeline import FeatureCache, RunSpec
from threshold_tuning_research.pipeline.datasets import dataset_names
from spikinn.evaluation.torch_logistic import TorchLogisticRegression
from spikinn.evaluation.torch_svc import TorchLinearSVC


def make_classifier(name: str):
    if name == "svc":
        return TorchLinearSVC(C=1.0)
    if name == "logistic":
        return TorchLogisticRegression(C=1.0)
    raise ValueError(f"unknown classifier {name!r}")


def build_X(cache: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    F_n, _, N, pool = cache.shape
    return cache[np.arange(F_n), offsets].transpose(1, 0, 2).reshape(N, F_n * pool)


def compute_raw_feature_gradient(clf) -> np.ndarray:
    coef, _ = clf.loss_state()
    grad_scaled = (coef @ clf._Wa.T)[:, :-1]  # (N, D)
    grad_raw = grad_scaled / (clf._feat_range_t.unsqueeze(0) + 1e-12)
    return grad_raw.cpu().numpy()


def gradient_at_external_features(clf, X_raw: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_scaled = clf._scale_np(X_raw.astype(np.float32, copy=True))
    X_t = torch.from_numpy(X_scaled).to(clf._device)
    n = X_t.shape[0]
    Xa = torch.cat([X_t, torch.ones(n, 1, device=clf._device)], dim=1)
    K = clf._K
    y_t = torch.as_tensor(y, dtype=torch.long, device=clf._device)
    Y = -torch.ones(n, K, device=clf._device)
    Y[torch.arange(n, device=clf._device), y_t] = 1.0

    Wa = clf._Wa
    C = clf._C
    score = Xa @ Wa
    margin = Y * score
    if clf._loss_type == "svc":
        active = (margin < 1.0).float()
        coef = -2.0 * C * active * (1.0 - margin) * Y
    else:
        sigma = torch.sigmoid(margin)
        coef = -C * (1.0 - sigma) * Y

    grad_scaled = (coef @ Wa.T)[:, :-1]
    grad_raw = grad_scaled / (clf._feat_range_t.unsqueeze(0) + 1e-12)
    return grad_raw.cpu().numpy()


def bootstrap_averaged_gradient(
    train_cache_offsets: np.ndarray,
    X_tr: np.ndarray,
    y_train: np.ndarray,
    classifier_name: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    grads = []
    n = X_tr.shape[0]
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)  # with replacement
        clf_k = make_classifier(classifier_name)
        clf_k.fit(X_tr[idx], y_train[idx])
        grads.append(gradient_at_external_features(clf_k, X_tr, y_train))
    return np.mean(grads, axis=0)


def find_best_offsets_linear(
    train_cache: np.ndarray,
    grad_raw: np.ndarray,
    current_offsets: np.ndarray | None = None,
    max_step: int = 0,
) -> np.ndarray:
    F_n, num_fracs, N, pool = train_cache.shape
    grad_per_neuron = grad_raw.reshape(N, F_n, pool).transpose(1, 0, 2)  # (F, N, pool)
    pseudo_score = np.einsum(
        "ifnp,inp->if", train_cache, grad_per_neuron, optimize=True
    )
    if max_step <= 0 or current_offsets is None:
        return pseudo_score.argmin(axis=1)
    # Mask out offsets outside the per-neuron trust region
    f_idx = np.arange(num_fracs)
    distance = np.abs(f_idx[None, :] - current_offsets[:, None])  # (F, num_fracs)
    masked = np.where(distance <= max_step, pseudo_score, np.inf)
    return masked.argmin(axis=1)


def find_best_offsets_quadratic(
    clf,
    train_cache: np.ndarray,
    current_offsets: np.ndarray,
    max_step: int = 0,
) -> np.ndarray:
    coef, hess_weight = clf.loss_state()
    grad_scaled = (coef @ clf._Wa.T)[:, :-1]  # (N, D)

    grad_scaled_np = grad_scaled.cpu().numpy()
    W_np = clf._Wa[:-1, :].cpu().numpy()  # (D, K)
    Hw_np = hess_weight.cpu().numpy()  # (N, K)
    range_np = clf._feat_range_t.cpu().numpy()  # (D,)

    F_n, num_fracs, N, pool = train_cache.shape
    new_offsets = np.empty(F_n, dtype=np.int64)

    for i in range(F_n):
        col_lo, col_hi = i * pool, (i + 1) * pool
        range_i = range_np[col_lo:col_hi] + 1e-12
        W_block = W_np[col_lo:col_hi, :]
        grad_i = grad_scaled_np[:, col_lo:col_hi]

        raw = train_cache[i]
        scaled = raw / range_i
        cur = scaled[current_offsets[i]]
        delta = scaled - cur  # (num_fracs, N, pool)

        linear = np.einsum("fnp,np->f", delta, grad_i, optimize=True)
        delta_s = np.einsum("fnp,pk->fnk", delta, W_block, optimize=True)
        quad = np.einsum("nk,fnk,fnk->f", Hw_np, delta_s, delta_s, optimize=True)
        dL = linear + quad

        if max_step > 0:
            lo = max(0, current_offsets[i] - max_step)
            hi = min(num_fracs - 1, current_offsets[i] + max_step)
            mask = np.full(num_fracs, np.inf)
            mask[lo : hi + 1] = 0.0
            dL = dL + mask

        new_offsets[i] = int(np.argmin(dL))

    return new_offsets


@dataclass
class AMTrajectory:
    history: list[dict]  # per-iter: iteration, train, test, n_changed, median_frac, time_s
    final_offsets: np.ndarray  # last committed offsets
    train_best_offsets: np.ndarray  # HEADLINE iterate (no peeking), ADR 0001
    best_test_offsets: np.ndarray  # test-peeking oracle only
    best_train: float
    best_train_test: float
    best_test: float


def run_alternating_minimization(
    train_cache: np.ndarray,
    y_train: np.ndarray,
    fractions: np.ndarray,
    *,
    test_cache: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    classifier: str = "logistic",
    mode: str = "linear",
    max_step: int = 2,
    max_iter: int = 15,
    bootstrap: int = 0,
    bootstrap_rng_seed: int = 42,
    verbose: bool = False,
) -> AMTrajectory:
    """Trust-region alternating minimization over per-neuron threshold offsets.

    The reusable headline algorithm, decoupled from CLI/IO/plotting. Per iteration:
    fit the surrogate read-out, take a trust-region first-order θ-step (Jacobi over all
    neurons), log accuracy. Reports the train-best iterate (ADR 0001); best_test is a
    test-peeking oracle. test_cache/y_test are optional (only used to log test accuracy).
    """
    fractions = np.asarray(fractions, dtype=np.float64)
    F_n = train_cache.shape[0]
    zero_idx = int(np.argmin(np.abs(fractions)))
    offsets = np.full(F_n, zero_idx, dtype=np.int64)
    has_test = test_cache is not None and y_test is not None

    history: list[dict] = []
    best_train = best_train_test = best_test = -1.0
    best_train_offsets = best_test_offsets = offsets.copy()

    for it in range(max_iter):
        t_iter = time.time()
        X_tr = build_X(train_cache, offsets)
        clf = make_classifier(classifier)
        clf.fit(X_tr, y_train)

        # Headline accuracy is always TorchLinearSVC; reuse it if it IS the surrogate.
        eval_clf = clf if classifier == "svc" else TorchLinearSVC(C=1.0).fit(X_tr, y_train)
        train_acc = float((eval_clf.predict(X_tr) == y_train).mean())
        if has_test:
            test_acc = float((eval_clf.predict(build_X(test_cache, offsets)) == y_test).mean())
        else:
            test_acc = float("nan")

        if mode == "linear":
            if bootstrap > 0:
                grad_raw = bootstrap_averaged_gradient(
                    offsets, X_tr, y_train, classifier,
                    n_bootstrap=bootstrap,
                    rng=np.random.default_rng(bootstrap_rng_seed + it),
                )
            else:
                grad_raw = compute_raw_feature_gradient(clf)
            new_offsets = find_best_offsets_linear(
                train_cache, grad_raw, current_offsets=offsets, max_step=max_step
            )
        else:
            new_offsets = find_best_offsets_quadratic(clf, train_cache, offsets, max_step=max_step)
        n_changed = int((new_offsets != offsets).sum())

        history.append({
            "iteration": it,
            "train": train_acc,
            "test": test_acc,
            "n_changed": n_changed,
            "median_frac": float(np.median(fractions[offsets])),
            "time_s": time.time() - t_iter,
        })
        if train_acc > best_train:
            best_train, best_train_test = train_acc, test_acc
            best_train_offsets = offsets.copy()
        if has_test and test_acc > best_test:
            best_test = test_acc
            best_test_offsets = offsets.copy()

        if verbose:
            print(
                f"  iter {it:2d}  train={train_acc:.4f}  test={test_acc:.4f}  "
                f"n_changed={n_changed:4d}  median_f={np.median(fractions[offsets]):+.3f}"
            )

        if n_changed == 0:
            if verbose:
                print("  converged")
            break
        offsets = new_offsets

    return AMTrajectory(
        history=history,
        final_offsets=offsets,
        train_best_offsets=best_train_offsets,
        best_test_offsets=best_test_offsets,
        best_train=best_train,
        best_train_test=best_train_test,
        best_test=best_test,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fashion_mnist", choices=dataset_names())
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-filters", type=int, default=256)
    parser.add_argument("--t-obj", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=15)
    parser.add_argument(
        "--max-step",
        type=int,
        default=0,
        help="Trust-region cap on per-neuron offset moves per iteration. "
        "0 = unlimited (current default). Applies to both linear and quadratic.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Number of bootstrap classifier fits per iteration to average the "
        "gradient over. 0 = single classifier (current default). K=5 is a "
        "reasonable starting point; trades 5× per-iter cost for lower-variance "
        "gradient direction. Only used by --mode linear.",
    )
    parser.add_argument(
        "--bootstrap-rng-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--mode",
        default="linear",
        choices=["linear", "quadratic"],
        help="ΔL approximation: linear (first-order, boundary-snapping) "
        "or quadratic (Newton with stale active set, interior optimum).",
    )
    parser.add_argument(
        "--classifier",
        default="svc",
        choices=["svc", "logistic"],
        help="Surrogate classifier for the optimisation loop. Final headline "
        "accuracy is always reported via TorchLinearSVC regardless.",
    )
    parser.add_argument("--cache-name", default="feature_cache_step0.05_drift0.75.pt")
    parser.add_argument("--out-name", default="alternating_minimization")
    args = parser.parse_args()

    set_seed(args.seed)
    t_obj_default = {"fashion_mnist": 0.85, "cifar10": 0.70}[args.dataset]
    t_obj = args.t_obj if args.t_obj is not None else t_obj_default

    spec = RunSpec.single(args.dataset, args.num_filters, t_obj, args.seed)
    cache_path = spec.model_dir / args.cache_name
    variant_tag = f"{args.classifier}_{args.mode}"
    if args.max_step > 0:
        variant_tag += f"_step{args.max_step}"
    if args.bootstrap > 0:
        variant_tag += f"_boot{args.bootstrap}"
    out_dir = spec.refinement_dir(args.out_name, variant_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.dataset} seed={args.seed} t_obj={t_obj}]  out={out_dir}")
    print(f"  cache = {cache_path}")

    fc = FeatureCache.load(cache_path)
    train_cache, test_cache = fc.train_cache, fc.test_cache
    y_train, y_test = fc.y_train, fc.y_test
    fractions = np.asarray(fc.perturbation_fractions, dtype=np.float64)

    traj = run_alternating_minimization(
        train_cache,
        y_train,
        fractions,
        test_cache=test_cache,
        y_test=y_test,
        classifier=args.classifier,
        mode=args.mode,
        max_step=args.max_step,
        max_iter=args.max_iter,
        bootstrap=args.bootstrap,
        bootstrap_rng_seed=args.bootstrap_rng_seed,
        verbose=True,
    )
    history = traj.history
    offsets = traj.final_offsets
    best_train, best_train_test, best_test = traj.best_train, traj.best_train_test, traj.best_test
    best_train_offsets, best_test_offsets = traj.train_best_offsets, traj.best_test_offsets

    # Final eval at final offsets (re-fit for stability)
    X_tr = build_X(train_cache, offsets)
    X_te = build_X(test_cache, offsets)
    clf = TorchLinearSVC(C=1.0)
    clf.fit(X_tr, y_train)
    final_train = float((clf.predict(X_tr) == y_train).mean())
    final_test = float((clf.predict(X_te) == y_test).mean())

    base = history[0]
    print(f"\nBaseline (iter 0): train={base['train']:.4f}  test={base['test']:.4f}")
    print(
        f"Final iter:        train={final_train:.4f}  test={final_test:.4f}  "
        f"Δtrain={final_train - base['train']:+.4f}  Δtest={final_test - base['test']:+.4f}"
    )
    print(
        f"Train-best iter:   train={best_train:.4f}  test={best_train_test:.4f}  "
        f"Δtest_vs_baseline={best_train_test - base['test']:+.4f}  [HEADLINE, no peeking]"
    )
    print(
        f"Best-test iter:    test={best_test:.4f}  "
        f"Δtest_vs_baseline={best_test - base['test']:+.4f}  [oracle, test peeking]"
    )

    with open(out_dir / "results.json", "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "t_obj": t_obj,
                "baseline": {"train": base["train"], "test": base["test"]},
                "final": {"train": final_train, "test": final_test},
                "train_best": {"train": best_train, "test": best_train_test},
                "best_test": best_test,
                "n_iterations": len(history),
                "history": history,
                "final_offsets": offsets.tolist(),
                "final_fractions": fractions[offsets].tolist(),
                "train_best_offsets": best_train_offsets.tolist(),
                "best_test_offsets": best_test_offsets.tolist(),
            },
            f,
            indent=2,
        )

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    its = [h["iteration"] for h in history]
    axes[0].plot(its, [h["train"] for h in history], "o-", label="train")
    axes[0].plot(its, [h["test"] for h in history], "s-", label="test")
    axes[0].axhline(base["train"], color="C0", ls="--", alpha=0.4)
    axes[0].axhline(base["test"], color="C1", ls="--", alpha=0.4)
    axes[0].set_ylabel("accuracy")
    axes[0].legend()
    axes[0].set_title(
        f"Alternating-minimization | {args.dataset} seed={args.seed}  "
        f"Δtest={final_test - base['test']:+.4f} (best={best_test - base['test']:+.4f})"
    )

    axes[1].plot(its, [h["n_changed"] for h in history], "o-")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("# neurons changed")
    axes[1].set_yscale("symlog")

    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_dir}/results.json + convergence.png")


if __name__ == "__main__":
    main()
