import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from applications.datasets import DATASETS

from .config import (
    T_OBJECTIVES,
    SEEDS,
    load_feature_group,
)
from .threshold_predictor import (
    build_feature_matrix,
    feature_group_ablation,
    fit_and_evaluate,
)


def _plot_top_features_vs_target(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    importance: dict[str, float],
    output_dir: str,
    top_n: int = 6,
) -> None:
    """Scatter plots of the top-N most important features vs optimal thresholds."""
    sorted_feats = sorted(importance.items(), key=lambda x: -x[1])[:top_n]
    top_names = [name for name, _ in sorted_feats]
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    n_cols = 3
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, feat_name in enumerate(top_names):
        ax = axes[i]
        idx = name_to_idx[feat_name]
        x = X[:, idx]
        rho, p_val = spearmanr(x, y)

        ax.scatter(x, y, alpha=0.15, s=8, edgecolors="none")
        # Linear trend line
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 2:
            z = np.polyfit(x[mask], y[mask], 1)
            x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), "r-", linewidth=1.5)

        short_name = feat_name.split(".")[-1]
        ax.set_xlabel(short_name, fontsize=10)
        ax.set_ylabel("optimal threshold", fontsize=10)
        ax.set_title(
            f"{short_name}\n"
            f"ρ={rho:.3f} (p={p_val:.1e})  imp={importance[feat_name]:.3f}",
            fontsize=9,
        )

    # Hide unused axes
    for j in range(len(top_names), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top Features vs Optimal Thresholds", fontsize=13, y=1.01)
    fig.tight_layout()
    path = f"{output_dir}/top_features_vs_threshold.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"\nSaved figure: {path}")


def run(
    dataset: str,
    *,
    seeds: list[int] | None = None,
):
    if seeds is None:
        seeds = SEEDS

    base_dir = f"logs/{dataset}/threshold_prediction"

    tqdm.write("=== Step 3: Prediction Analysis ===")
    tqdm.write(f"Scanning {base_dir} for completed conditions...")

    all_X = []
    all_y = []
    all_groups = []
    n_conditions = 0

    for t_obj in T_OBJECTIVES:
        tobj_dir = f"{base_dir}/tobj_{t_obj}"
        if not os.path.isdir(tobj_dir):
            continue

        for seed in seeds:
            feat_dir = f"{tobj_dir}/seed_{seed}/features"
            optimal_path = f"{feat_dir}/optimal_thresholds.npy"
            if not os.path.exists(optimal_path):
                continue

            # Load features (from step 1) and targets (from step 2)
            trajectory = load_feature_group(f"{feat_dir}/trajectory.json")
            distribution = load_feature_group(f"{feat_dir}/distribution.json")
            inter_neuron = load_feature_group(f"{feat_dir}/inter_neuron.json")
            optimal = np.load(optimal_path)

            X, names = build_feature_matrix(
                trajectory_features=trajectory,
                distribution_features=distribution,
                inter_neuron_features=inter_neuron,
            )

            all_X.append(X)
            all_y.append(optimal)
            all_groups.append(np.full(X.shape[0], seed))
            n_conditions += 1

    if not all_X:
        tqdm.write("No completed conditions found. Run steps 1 and 2 first.")
        return {}

    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    groups_combined = np.concatenate(all_groups)

    tqdm.write(
        f"Loaded {n_conditions} conditions: "
        f"{X_combined.shape[0]} neurons, {X_combined.shape[1]} features"
    )
    tqdm.write(f"Seeds: {sorted(set(int(g) for g in groups_combined))}")

    # Clean data
    X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Fit prediction models ---
    tqdm.write("\nFitting prediction models (leave-one-seed-out CV)...")
    model_results = fit_and_evaluate(X_combined, y_combined, groups_combined, names)

    tqdm.write("\nModel Results:")
    for name, res in model_results["per_model"].items():
        tqdm.write(
            f"  {name:6s}: R²={res['r2_mean']:.4f} ± {res['r2_std']:.4f}  "
            f"MAE={res['mae_mean']:.4f} ± {res['mae_std']:.4f}"
        )
    tqdm.write(f"\nBest model: {model_results['best_model_name']}")

    # --- Feature importance ---
    tqdm.write("\nTop 10 features by importance:")
    importance = model_results["feature_importance"]
    sorted_feats = sorted(importance.items(), key=lambda x: -x[1])
    for feat_name, imp in sorted_feats[:10]:
        tqdm.write(f"  {imp:+.4f}  {feat_name}")

    # --- Feature group ablation ---
    tqdm.write("\nFeature group ablation:")
    ablation = feature_group_ablation(X_combined, y_combined, groups_combined, names)

    for res in ablation:
        tqdm.write(
            f"  {res['group']:15s} ({res['n_features']:2d} feats): "
            f"R²={res['r2_mean']:.4f} ± {res['r2_std']:.4f}"
        )

    # --- Plot top features vs optimal thresholds ---
    analysis_dir = f"{base_dir}/prediction_analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    _plot_top_features_vs_target(
        X_combined, y_combined, names, importance, analysis_dir
    )

    # --- Save ---

    analysis_results = {
        "n_conditions": n_conditions,
        "n_neurons": int(X_combined.shape[0]),
        "n_features": int(X_combined.shape[1]),
        "feature_names": names,
        "model_results": {
            k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
            for k, v in model_results["per_model"].items()
        },
        "best_model": model_results["best_model_name"],
        "feature_importance": importance,
        "ablation": ablation,
    }
    with open(f"{analysis_dir}/results.json", "w") as f:
        json.dump(analysis_results, f, indent=4)

    tqdm.write(f"\nResults saved to {analysis_dir}/results.json")
    return analysis_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3: Find relationship between metrics and optimal thresholds"
    )
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--seeds", type=int, nargs="+", help="Only use these seeds")
    args = parser.parse_args()
    run(args.dataset, seeds=args.seeds)
