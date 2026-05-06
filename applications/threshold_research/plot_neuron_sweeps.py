"""Plot per-neuron accuracy curves across all cached perturbation levels.

For each neuron, sweeps its threshold from -50% to +50% while keeping all
other neurons at their original thresholds. Shows whether the landscape
has local minima or is smooth/convex per neuron.
"""

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from spiking.evaluation.ridge_column_swap import RidgeColumnSwap

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-neuron threshold sweeps")
    parser.add_argument("--cache-path", required=True)
    parser.add_argument(
        "--n-neurons", type=int, default=8, help="Top N neurons to plot"
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    logger.info("Loading cache...")
    cache = torch.load(args.cache_path, weights_only=False)
    train_cache = cache["train_cache"]
    test_cache = cache["test_cache"]
    y_train, y_test = cache["y_train"], cache["y_test"]
    fractions = cache["perturbation_fractions"]
    pool_dim = cache["pool_size"] ** 2
    F = train_cache.shape[0]
    zero_idx = fractions.index(0.0)

    # Build baseline features
    N_train, N_test = train_cache.shape[2], test_cache.shape[2]
    X_train = np.empty((N_train, F * pool_dim), dtype=np.float32)
    X_test = np.empty((N_test, F * pool_dim), dtype=np.float32)
    for f in range(F):
        X_train[:, f * pool_dim : (f + 1) * pool_dim] = train_cache[f, zero_idx]
        X_test[:, f * pool_dim : (f + 1) * pool_dim] = test_cache[f, zero_idx]

    clf = RidgeColumnSwap(alpha=args.alpha)
    clf.fit(X_train, y_train)
    baseline_val = float((clf.predict(X_test) == y_test).mean())
    baseline_train = float((clf.predict(X_train) == y_train).mean())
    logger.info("Baseline — train: %.4f, val: %.4f", baseline_train, baseline_val)

    # Neuron importance
    coef_imp = np.abs(clf.weights).sum(axis=1)
    neuron_imp = np.array(
        [coef_imp[f * pool_dim : (f + 1) * pool_dim].sum() for f in range(F)]
    )
    top_neurons = np.argsort(neuron_imp)[-args.n_neurons :][::-1]

    # Sweep each neuron
    n_cols = 4
    n_rows = (args.n_neurons + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for plot_idx, neuron in enumerate(top_neurons):
        ax = axes_flat[plot_idx]
        col_start = neuron * pool_dim
        col_indices = list(range(col_start, col_start + pool_dim))

        train_accs, val_accs = [], []
        for level_idx in range(len(fractions)):
            new_train_cols = train_cache[neuron, level_idx]

            X_val_mod = X_test.copy()
            X_val_mod[:, col_start : col_start + pool_dim] = test_cache[
                neuron, level_idx
            ]
            X_train_mod = X_train.copy()
            X_train_mod[:, col_start : col_start + pool_dim] = new_train_cols

            y_pred_val = clf.predict_swapped(col_indices, new_train_cols, X_val_mod)
            y_pred_train = clf.predict_swapped(col_indices, new_train_cols, X_train_mod)
            val_accs.append(float((y_pred_val == y_test).mean()))
            train_accs.append(float((y_pred_train == y_train).mean()))

        frac_pct = [f * 100 for f in fractions]
        ax.plot(
            frac_pct, [a * 100 for a in train_accs], "b-o", markersize=3, label="Train"
        )
        ax.plot(frac_pct, [a * 100 for a in val_accs], "r-o", markersize=3, label="Val")
        ax.axhline(
            baseline_val * 100, color="green", linestyle="--", linewidth=0.8, alpha=0.5
        )
        ax.axhline(
            baseline_train * 100, color="blue", linestyle="--", linewidth=0.8, alpha=0.3
        )
        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

        best_val_idx = int(np.argmax(val_accs))
        best_train_idx = int(np.argmax(train_accs))
        ax.set_title(
            f"Neuron {neuron} (imp={neuron_imp[neuron]:.1f})\n"
            f"best val={fractions[best_val_idx] * 100:+.0f}%, "
            f"best train={fractions[best_train_idx] * 100:+.0f}%",
            fontsize=9,
        )
        ax.set_xlabel("Threshold perturbation (%)")
        ax.set_ylabel("Accuracy (%)")
        if plot_idx == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        logger.info(
            "Neuron %3d: val [%.4f, %.4f] best=%+.0f%%, train [%.4f, %.4f] best=%+.0f%%",
            neuron,
            min(val_accs),
            max(val_accs),
            fractions[best_val_idx] * 100,
            min(train_accs),
            max(train_accs),
            fractions[best_train_idx] * 100,
        )

    # Hide unused axes
    for i in range(len(top_neurons), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.suptitle(
        f"Per-Neuron Accuracy Sweep (top {args.n_neurons} by importance)",
        fontsize=12,
    )
    plt.tight_layout()
    output = args.output or args.cache_path.replace(".pt", "_neuron_sweeps.png")
    plt.savefig(output, dpi=150)
    logger.info("Saved to %s", output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
