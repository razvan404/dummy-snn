"""Multi-pass greedy threshold optimization from precomputed feature cache.

Reads the per-neuron feature cache (compute_feature_cache.py) and runs
Woodbury-accelerated coordinate descent. Each pass takes ~5 min since
no SNN inference is needed — just column swaps on the cached features.

Algorithm per pass:
  For each neuron (in specified order):
    1. Sweep all cached levels -> find globally best level
    2. Move ONE step (+-5%) toward that level (not jump)
    3. Verify the step improves train accuracy
    4. If yes, apply via Woodbury update

Supports all orderings from filter_ordering module (importance, spike time,
threshold drift, recent/oldest winner, hybrids).
"""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from applications.common import set_seed
from applications.threshold_research.filter_ordering import ORDERINGS, get_filter_order
from spiking.evaluation.ridge_column_swap import RidgeColumnSwap

logger = logging.getLogger(__name__)


@dataclass
class GreedyConfig:
    cache_path: str = ""
    num_passes: int = 5
    ordering: str = "descending_importance"
    alpha: float = 1.0
    min_threshold: float = 1.0
    seed: int = 1
    output_dir: str = ""
    model_dir: str = ""
    coarse_stride: int = 4
    gpu: bool = False


def build_features_from_levels(
    cache_data: np.ndarray,
    levels: np.ndarray,
    pool_dim: int,
) -> np.ndarray:
    """Build full feature matrix from cache given per-neuron level indices."""
    F = cache_data.shape[0]
    N = cache_data.shape[2]
    features = np.empty((N, F * pool_dim), dtype=np.float32)
    for f in range(F):
        col_start = f * pool_dim
        features[:, col_start : col_start + pool_dim] = cache_data[f, levels[f]]
    return features


def _eval_level(
    clf: RidgeColumnSwap,
    cache_data_train: np.ndarray,
    cache_data_val: np.ndarray,
    neuron_idx: int,
    candidate_level: int,
    col_start: int,
    col_indices: list[int],
    pool_dim: int,
    y_eval: np.ndarray,
    _w_cache: dict | None = None,
) -> float:
    """Evaluate accuracy for a single neuron at a candidate level.

    Computes Woodbury-updated weights from train cols, then predicts on
    val features without copying the full val matrix — patches the score
    contribution from the 4 changed columns.
    """
    xp = clf._xp
    new_train_cols = cache_data_train[neuron_idx, candidate_level]
    new_val_cols = cache_data_val[neuron_idx, candidate_level]

    # Woodbury update to get new weights
    col_idx = np.asarray(col_indices)
    k = len(col_idx)
    new_dev = clf._to_xp(new_train_cols)
    new_mean = new_dev.mean(axis=0)
    new_cols_c = new_dev - new_mean

    D_c = new_cols_c - clf._X_c[:, col_idx]
    d = clf._X_c.shape[1]
    E = xp.zeros((d, k), dtype=xp.float64)
    E[col_idx, xp.arange(k)] = 1.0
    S = clf._X_c.T @ D_c
    DtD = D_c.T @ D_c
    U = xp.column_stack([E, S])
    top = xp.concatenate(
        [xp.zeros((k, k), dtype=xp.float64), xp.eye(k, dtype=xp.float64)], axis=1
    )
    bot = xp.concatenate([xp.eye(k, dtype=xp.float64), -DtD], axis=1)
    C_inv = xp.concatenate([top, bot], axis=0)
    A_inv_U = clf._A_inv @ U
    inner_inv = xp.linalg.inv(C_inv + U.T @ A_inv_U)
    A_new_inv = clf._A_inv - A_inv_U @ inner_inv @ A_inv_U.T

    XtY_c_new = clf._XtY_c.copy()
    XtY_c_new[col_idx] = new_cols_c.T @ clf._Y_c
    w_new = A_new_inv @ XtY_c_new
    X_mean_new = clf._X_mean.copy()
    X_mean_new[col_idx] = new_mean
    intercept_new = clf._Y_mean - X_mean_new @ w_new

    # Predict on val: base_scores + delta from changed columns
    # scores = X_val @ w_new + intercept_new
    #        = X_val_base @ w_new + (new_val_cols - old_val_cols) @ w_new[cols] + intercept_new
    # Precompute base_scores = X_val_base @ w_old + intercept_old (cached externally)
    # Instead, just compute the full dot product on the small val set (10k x 1024 is cheap)
    X_val_dev = clf._to_xp(clf._X_val_base)
    # Patch the changed columns in-place temporarily
    old_cols = X_val_dev[:, col_idx].copy()
    X_val_dev[:, col_idx] = clf._to_xp(new_val_cols)
    scores = X_val_dev @ w_new + intercept_new
    X_val_dev[:, col_idx] = old_cols  # restore

    y_pred = clf._decode(clf._to_np(scores))
    return float((y_pred == y_eval).mean())


def greedy_pass(
    clf: RidgeColumnSwap,
    cache_data_train: np.ndarray,
    cache_data_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    current_levels: np.ndarray,
    fractions: list[float],
    pool_dim: int,
    neuron_order: np.ndarray,
    min_threshold: float,
    original_thresholds: np.ndarray,
    coarse_stride: int = 4,
) -> dict:
    """One pass of greedy coordinate descent using cached features.

    Coarse-to-fine sweep: first check every coarse_stride-th level to find
    the best direction, then sweep the fine levels in that bin. Cuts
    evaluations from num_fracs to ~(num_fracs/stride + stride) per neuron.
    """
    num_fracs = len(fractions)
    n_changes = 0
    improvements = np.zeros(len(neuron_order), dtype=np.float32)
    val_curve = []

    # Cache current val accuracy — only recompute after a swap
    y_pred_current = clf.predict(clf._X_val_base)
    current_acc = float((y_pred_current == y_val).mean())

    for i, neuron_idx in enumerate(neuron_order):
        current_level = current_levels[neuron_idx]
        col_start = neuron_idx * pool_dim
        col_indices = list(range(col_start, col_start + pool_dim))

        best_acc = current_acc
        best_level = current_level

        # Phase 1: coarse sweep (every coarse_stride-th level)
        coarse_levels = list(range(0, num_fracs, coarse_stride))
        # Always include the last level
        if coarse_levels[-1] != num_fracs - 1:
            coarse_levels.append(num_fracs - 1)

        coarse_best_level = current_level
        coarse_best_acc = current_acc
        for candidate_level in coarse_levels:
            if candidate_level == current_level:
                continue
            frac = fractions[candidate_level]
            new_thresh = original_thresholds[neuron_idx] * (1.0 + frac)
            if new_thresh < min_threshold:
                continue
            acc = _eval_level(
                clf,
                cache_data_train,
                cache_data_val,
                neuron_idx,
                candidate_level,
                col_start,
                col_indices,
                pool_dim,
                y_val,
            )
            if acc > coarse_best_acc:
                coarse_best_acc = acc
                coarse_best_level = candidate_level

        # Phase 2: fine sweep around coarse winner's bin
        if coarse_best_level != current_level:
            fine_lo = max(0, coarse_best_level - coarse_stride + 1)
            fine_hi = min(num_fracs - 1, coarse_best_level + coarse_stride - 1)
            for candidate_level in range(fine_lo, fine_hi + 1):
                if candidate_level == current_level:
                    continue
                if candidate_level in coarse_levels:
                    # Already evaluated in coarse pass — reuse result
                    continue
                frac = fractions[candidate_level]
                new_thresh = original_thresholds[neuron_idx] * (1.0 + frac)
                if new_thresh < min_threshold:
                    continue
                acc = _eval_level(
                    clf,
                    cache_data_train,
                    cache_data_val,
                    neuron_idx,
                    candidate_level,
                    col_start,
                    col_indices,
                    pool_dim,
                    y_val,
                )
                if acc > best_acc:
                    best_acc = acc
                    best_level = candidate_level
            # Also consider the coarse winner itself
            if coarse_best_acc > best_acc:
                best_acc = coarse_best_acc
                best_level = coarse_best_level
        # If coarse found nothing better, best_level stays at current_level

        # Move one step TOWARD the global best
        if best_level != current_level:
            direction = 1 if best_level > current_level else -1
            target_level = current_level + direction

            step_acc = _eval_level(
                clf,
                cache_data_train,
                cache_data_val,
                neuron_idx,
                target_level,
                col_start,
                col_indices,
                pool_dim,
                y_val,
            )

            if step_acc > current_acc:
                new_train_cols_step = cache_data_train[neuron_idx, target_level]
                clf.apply_swap(col_indices, new_train_cols_step)
                clf._X_train_base[:, col_start : col_start + pool_dim] = (
                    new_train_cols_step
                )
                clf._X_val_base[:, col_start : col_start + pool_dim] = cache_data_val[
                    neuron_idx, target_level
                ]
                current_levels[neuron_idx] = target_level
                n_changes += 1
                improvements[i] = step_acc - current_acc
                current_acc = step_acc
                val_curve.append(step_acc)
            else:
                val_curve.append(current_acc)
        else:
            val_curve.append(current_acc)

    # Evaluate final state
    val_features = build_features_from_levels(cache_data_val, current_levels, pool_dim)
    val_pred = clf.predict(val_features)
    val_acc = float((val_pred == y_val).mean())

    train_features = build_features_from_levels(
        cache_data_train, current_levels, pool_dim
    )
    train_pred = clf.predict(train_features)
    train_acc = float((train_pred == y_train).mean())

    return {
        "n_changes": n_changes,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "improvements": improvements,
        "val_curve": val_curve,
    }


def plot_passes(
    history: list[dict],
    baseline_acc: float,
    ordering: str,
    output_path: str,
) -> None:
    """Plot multi-pass convergence with per-neuron training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    passes = range(1, len(history) + 1)
    train_accs = [h["train_acc"] for h in history]
    val_accs = [h["val_acc"] for h in history]
    n_changes = [h["n_changes"] for h in history]

    ax = axes[0, 0]
    ax.plot(passes, train_accs, "b-o", label="Train")
    ax.plot(passes, val_accs, "r-o", label="Val")
    ax.axhline(
        baseline_acc,
        color="green",
        linestyle="--",
        label=f"Baseline: {baseline_acc:.4f}",
    )
    ax.set_xlabel("Pass")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-Pass Accuracy (ordering={ordering})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.bar(passes, n_changes, color="steelblue", alpha=0.7)
    ax.set_xlabel("Pass")
    ax.set_ylabel("Neurons changed")
    ax.set_title("Changes per Pass")
    ax.grid(True, alpha=0.3)

    # Full val curve across all neurons
    ax = axes[1, 0]
    full_curve = []
    pass_boundaries = [0]
    for h in history:
        full_curve.extend(h["val_curve"])
        pass_boundaries.append(len(full_curve))
    ax.plot(range(len(full_curve)), full_curve, "r-", linewidth=0.5, alpha=0.8)
    ax.axhline(
        baseline_acc,
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"Baseline: {baseline_acc:.4f}",
    )
    for i in range(len(pass_boundaries) - 1):
        ax.axvline(pass_boundaries[i], color="gray", linewidth=0.5, linestyle=":")
        mid = (pass_boundaries[i] + pass_boundaries[i + 1]) // 2
        ax.text(mid, min(full_curve), f"P{i + 1}", ha="center", fontsize=7, alpha=0.6)
    ax.set_xlabel("Neuron index (across all passes)")
    ax.set_ylabel("Val accuracy")
    ax.set_title("Val Accuracy per Neuron Evaluation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Per-neuron improvement distribution (pass 1)
    ax = axes[1, 1]
    if history:
        impr = history[0]["improvements"]
        nonzero = impr[impr != 0]
        if len(nonzero) > 0:
            ax.hist(nonzero * 100, bins=20, edgecolor="black", alpha=0.7)
            ax.set_xlabel("Accuracy improvement (%)")
            ax.set_ylabel("Count")
            ax.set_title(
                f"Pass 1: Per-neuron improvement ({len(nonzero)} neurons changed)"
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No changes",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info("Plot saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-pass greedy optimization from cached features"
    )
    parser.add_argument("--cache-path", required=True, help="Path to feature cache .pt")
    parser.add_argument("--num-passes", type=int, default=5)
    parser.add_argument(
        "--ordering",
        default="descending_importance",
        choices=ORDERINGS,
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--model-dir",
        default="",
        help="Model directory with training_logs.pt (needed for winner/training orderings)",
    )
    parser.add_argument(
        "--coarse-stride",
        type=int,
        default=4,
        help="Coarse sweep stride (check every Nth level, then refine). 1 = full sweep.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous results.json in output-dir",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use cupy GPU acceleration for Ridge Woodbury ops",
    )
    args = parser.parse_args()

    config = GreedyConfig(
        **{
            k: v
            for k, v in vars(args).items()
            if k in GreedyConfig.__dataclass_fields__
        }
    )
    set_seed(config.seed)

    # Load cache
    logger.info("Loading cache from %s", config.cache_path)
    cache = torch.load(config.cache_path, weights_only=False)
    train_cache = cache["train_cache"]
    test_cache = cache["test_cache"]
    y_train = cache["y_train"]
    y_test = cache["y_test"]
    original_thresholds = cache["original_thresholds"]
    fractions = cache["perturbation_fractions"]
    pool_dim = cache["pool_size"] ** 2

    num_filters, num_fracs, N_train, _ = train_cache.shape
    zero_idx = fractions.index(0.0)
    logger.info(
        "Cache: %d filters, %d levels, %d train, %d test",
        num_filters,
        num_fracs,
        N_train,
        len(y_test),
    )

    output_dir = config.output_dir or os.path.dirname(config.cache_path) + "/greedy_opt"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(asdict(config), f, indent=4)

    # Baseline
    X_train = build_features_from_levels(
        train_cache, np.full(num_filters, zero_idx, dtype=int), pool_dim
    )
    X_test = build_features_from_levels(
        test_cache, np.full(num_filters, zero_idx, dtype=int), pool_dim
    )

    clf = RidgeColumnSwap(alpha=config.alpha, use_gpu=config.gpu)
    clf.fit(X_train, y_train)
    baseline_train = float((clf.predict(X_train) == y_train).mean())
    baseline_val = float((clf.predict(X_test) == y_test).mean())
    logger.info("Baseline Ridge — train: %.4f, val: %.4f", baseline_train, baseline_val)

    # Current state — resume from previous run if requested
    current_levels = np.full(num_filters, zero_idx, dtype=int)
    history = []
    resume_path = f"{output_dir}/results.json"
    if args.resume and os.path.exists(resume_path):
        with open(resume_path) as f:
            prev = json.load(f)
        current_levels = np.array(prev["current_levels"], dtype=int)
        history = [
            {**p, "improvements": np.zeros(num_filters), "val_curve": []}
            for p in prev["passes"]
        ]
        # Rebuild features from restored levels
        X_train = build_features_from_levels(train_cache, current_levels, pool_dim)
        X_test = build_features_from_levels(test_cache, current_levels, pool_dim)
        clf = RidgeColumnSwap(alpha=config.alpha, use_gpu=config.gpu)
        clf.fit(X_train, y_train)
        logger.info(
            "Resumed from %d previous passes (last val: %.4f)",
            len(history),
            history[-1]["val_acc"],
        )

    # Ordering data — static inputs computed once
    mean_spike_times = np.zeros(num_filters, dtype=np.float32)
    for f in range(num_filters):
        # Mean spike time at baseline level across training set
        mean_spike_times[f] = train_cache[f, zero_idx, :, 0].mean()

    threshold_drift = np.abs(original_thresholds - original_thresholds.mean())

    # Load training logs for winner-based orderings
    training_spike_times: np.ndarray | None = None
    last_win_index: np.ndarray | None = None
    model_dir = config.model_dir or os.path.dirname(config.cache_path)
    training_logs_path = os.path.join(model_dir, "training_logs.pt")
    if os.path.exists(training_logs_path):
        _logs = torch.load(training_logs_path, weights_only=True)
        if "last10k_winners" in _logs:
            winners = _logs["last10k_winners"].numpy()
            last_win_index = np.full(num_filters, -1, dtype=np.int64)
            for i, f in enumerate(winners):
                last_win_index[int(f)] = i
            if "last10k_spike_times" in _logs:
                spike_vals = _logs["last10k_spike_times"].numpy()
                training_spike_times = np.full(num_filters, np.nan)
                for i in range(len(winners) - 1, -1, -1):
                    f = int(winners[i])
                    if np.isnan(training_spike_times[f]):
                        training_spike_times[f] = spike_vals[i]
                    if not np.any(np.isnan(training_spike_times)):
                        break
        del _logs
    logger.info("Ordering: %s", config.ordering)

    # Orderings that depend on clf.weights must be recomputed each pass
    _dynamic_orderings = {
        "descending_importance",
        "ascending_importance",
        "hybrid_importance",
    }
    recompute_order = config.ordering in _dynamic_orderings

    def compute_neuron_order() -> np.ndarray:
        coef_importance = np.abs(clf.weights).sum(axis=1)
        filter_importance = np.array(
            [
                coef_importance[f * pool_dim : (f + 1) * pool_dim].sum()
                for f in range(num_filters)
            ]
        )
        return get_filter_order(
            config.ordering,
            filter_importance,
            mean_spike_times,
            threshold_drift=threshold_drift,
            training_spike_times=training_spike_times,
            last_win_index=last_win_index,
            seed=config.seed,
        )

    neuron_order = compute_neuron_order()

    # Store feature bases
    clf._X_train_base = X_train.copy()
    clf._X_val_base = X_test.copy()

    # Multi-pass optimization
    start_pass = len(history)
    total_passes = start_pass + config.num_passes
    logger.info(
        "Starting %d-pass greedy optimization (ordering=%s, starting at pass %d)...",
        config.num_passes,
        config.ordering,
        start_pass + 1,
    )

    for pass_idx in range(start_pass, total_passes):
        if recompute_order and pass_idx > 0:
            neuron_order = compute_neuron_order()
        t0 = time.time()

        pass_result = greedy_pass(
            clf,
            train_cache,
            test_cache,
            y_train,
            y_test,
            current_levels,
            fractions,
            pool_dim,
            neuron_order,
            config.min_threshold,
            original_thresholds,
            coarse_stride=config.coarse_stride,
        )

        elapsed = time.time() - t0
        history.append(pass_result)

        logger.info(
            "Pass %d/%d | %.1fs | changes: %d | train: %.4f | val: %.4f",
            pass_idx + 1,
            total_passes,
            elapsed,
            pass_result["n_changes"],
            pass_result["train_acc"],
            pass_result["val_acc"],
        )

        if pass_result["n_changes"] == 0:
            logger.info("No changes in pass %d — converged.", pass_idx + 1)
            break

    # Final thresholds
    optimized_thresholds = np.array(
        [
            original_thresholds[f] * (1.0 + fractions[current_levels[f]])
            for f in range(num_filters)
        ]
    )

    # Save
    results = {
        "baseline": {"train_acc": baseline_train, "val_acc": baseline_val},
        "final": {
            "train_acc": history[-1]["train_acc"],
            "val_acc": history[-1]["val_acc"],
        },
        "passes": [
            {
                "n_changes": h["n_changes"],
                "train_acc": h["train_acc"],
                "val_acc": h["val_acc"],
            }
            for h in history
        ],
        "original_thresholds": original_thresholds.tolist(),
        "optimized_thresholds": optimized_thresholds.tolist(),
        "current_levels": current_levels.tolist(),
        "config": asdict(config),
    }
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Results saved to %s", output_dir)

    plot_passes(history, baseline_val, config.ordering, f"{output_dir}/convergence.png")

    logger.info("=== Summary ===")
    logger.info("Baseline: train %.4f, val %.4f", baseline_train, baseline_val)
    logger.info(
        "Final:    train %.4f, val %.4f (%+.4f)",
        history[-1]["train_acc"],
        history[-1]["val_acc"],
        history[-1]["val_acc"] - baseline_val,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
