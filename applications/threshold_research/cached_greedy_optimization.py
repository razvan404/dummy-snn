"""Multi-pass greedy threshold optimization from precomputed feature cache.

Reads the per-neuron feature cache (compute_feature_cache.py) and runs
Woodbury-accelerated coordinate descent. Each pass takes ~5 min since
no SNN inference is needed — just column swaps on the cached features.

Algorithm per pass:
  For each neuron (in specified order):
    1. Sweep all cached levels → find globally best level
    2. Move ONE step (±5%) toward that level (not jump)
    3. Verify the step improves train accuracy
    4. If yes, apply via Woodbury update

Supports configurable ordering: descending_importance, ascending_importance,
random, forward (neuron index), reverse.
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


def get_neuron_order(
    ordering: str,
    clf: RidgeColumnSwap,
    num_filters: int,
    pool_dim: int,
    seed: int = 1,
) -> np.ndarray:
    """Compute neuron ordering based on strategy.

    :param ordering: One of descending_importance, ascending_importance,
        random, forward, reverse.
    :returns: (num_filters,) array of neuron indices.
    """
    if ordering in ("descending_importance", "ascending_importance"):
        coef_importance = np.abs(clf.weights).sum(axis=1)
        neuron_importance = np.array(
            [
                coef_importance[f * pool_dim : (f + 1) * pool_dim].sum()
                for f in range(num_filters)
            ]
        )
        order = np.argsort(neuron_importance)
        if ordering == "descending_importance":
            order = order[::-1]
        return order.copy()
    if ordering == "random":
        rng = np.random.RandomState(seed)
        order = np.arange(num_filters)
        rng.shuffle(order)
        return order
    if ordering == "forward":
        return np.arange(num_filters)
    if ordering == "reverse":
        return np.arange(num_filters)[::-1].copy()
    raise ValueError(f"Unknown ordering: {ordering}")


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
) -> dict:
    """One pass of greedy coordinate descent using cached features.

    For each neuron in order:
      1. Sweep all levels to find the globally best level
      2. Move one step toward it (not jump)
      3. Apply if the one-step move improves train accuracy
    """
    num_fracs = len(fractions)
    n_changes = 0
    improvements = np.zeros(len(neuron_order), dtype=np.float32)
    train_curve = []

    for i, neuron_idx in enumerate(neuron_order):
        current_level = current_levels[neuron_idx]
        col_start = neuron_idx * pool_dim
        col_indices = list(range(col_start, col_start + pool_dim))

        # Current accuracy
        y_pred_current = clf.predict(clf._X_train_base)
        current_acc = float((y_pred_current == y_train).mean())
        best_acc = current_acc
        best_level = current_level

        # Sweep ALL levels to find global optimum direction
        for candidate_level in range(num_fracs):
            if candidate_level == current_level:
                continue
            frac = fractions[candidate_level]
            new_thresh = original_thresholds[neuron_idx] * (1.0 + frac)
            if new_thresh < min_threshold:
                continue

            new_train_cols = cache_data_train[neuron_idx, candidate_level]
            X_train_mod = clf._X_train_base.copy()
            X_train_mod[:, col_start : col_start + pool_dim] = new_train_cols

            y_pred = clf.predict_swapped(col_indices, new_train_cols, X_train_mod)
            acc = float((y_pred == y_train).mean())
            if acc > best_acc:
                best_acc = acc
                best_level = candidate_level

        # Move one step TOWARD the global best
        if best_level != current_level:
            direction = 1 if best_level > current_level else -1
            target_level = current_level + direction

            new_train_cols_step = cache_data_train[neuron_idx, target_level]
            X_train_check = clf._X_train_base.copy()
            X_train_check[:, col_start : col_start + pool_dim] = new_train_cols_step
            y_pred_check = clf.predict_swapped(
                col_indices, new_train_cols_step, X_train_check
            )
            step_acc = float((y_pred_check == y_train).mean())

            if step_acc > current_acc:
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
                train_curve.append(step_acc)
            else:
                train_curve.append(current_acc)
        else:
            train_curve.append(current_acc)

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
        "train_curve": train_curve,
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

    # Full training curve
    ax = axes[1, 0]
    full_curve = []
    pass_boundaries = [0]
    for h in history:
        full_curve.extend(h["train_curve"])
        pass_boundaries.append(len(full_curve))
    ax.plot(range(len(full_curve)), full_curve, "b-", linewidth=0.5, alpha=0.8)
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
    ax.set_ylabel("Train accuracy")
    ax.set_title("Training Accuracy per Neuron Evaluation")
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
                0.5, 0.5, "No changes", ha="center", va="center", transform=ax.transAxes
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
        choices=[
            "descending_importance",
            "ascending_importance",
            "random",
            "forward",
            "reverse",
        ],
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="")
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

    clf = RidgeColumnSwap(alpha=config.alpha)
    clf.fit(X_train, y_train)
    baseline_train = float((clf.predict(X_train) == y_train).mean())
    baseline_val = float((clf.predict(X_test) == y_test).mean())
    logger.info("Baseline Ridge — train: %.4f, val: %.4f", baseline_train, baseline_val)

    # Current state
    current_levels = np.full(num_filters, zero_idx, dtype=int)

    # Neuron ordering
    neuron_order = get_neuron_order(
        config.ordering, clf, num_filters, pool_dim, config.seed
    )
    logger.info("Ordering: %s", config.ordering)

    # Store feature bases
    clf._X_train_base = X_train.copy()
    clf._X_val_base = X_test.copy()

    # Multi-pass optimization
    history = []
    logger.info(
        "Starting %d-pass greedy optimization (ordering=%s)...",
        config.num_passes,
        config.ordering,
    )

    for pass_idx in range(config.num_passes):
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
        )

        elapsed = time.time() - t0
        history.append(pass_result)

        logger.info(
            "Pass %d/%d | %.1fs | changes: %d | train: %.4f | val: %.4f",
            pass_idx + 1,
            config.num_passes,
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
