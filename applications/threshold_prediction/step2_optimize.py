import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from applications.datasets import DATASETS, create_dataset

from .analytical_baselines import evaluate_all_baselines
from .cached_coordinate_descent import CachedThresholdOptimizer
from .config import (
    T_OBJECTIVES,
    SEEDS,
    find_trained_models,
    load_feature_group,
)


def optimize_single(
    model_path: str,
    t_obj: float,
    seed: int,
    train_loader,
    val_loader,
    output_dir: str,
) -> dict:
    """Run per-neuron optimization and baselines for one trained model.

    Saves: optimization_results.json, baselines.json, features/optimal_thresholds.npy.
    """
    from spiking import load_model

    model = load_model(model_path)
    layer = model.layers[0]

    # Build optimizer
    tqdm.write(f"  tobj={t_obj} seed={seed}: building optimizer...")
    optimizer = CachedThresholdOptimizer.from_layer_and_data(
        layer, train_loader, val_loader, t_target=t_obj
    )

    # Per-neuron optimization
    trained_thresholds = layer.thresholds.detach().cpu().numpy()
    opt_result = optimizer.per_neuron_optimize(
        trained_thresholds, n_candidates=30, search_range=(0.5, 1.5)
    )
    optimal_thresholds = opt_result["optimal_thresholds"]

    # Save optimization results
    with open(f"{output_dir}/optimization_results.json", "w") as f:
        json.dump(
            {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in opt_result.items()
            },
            f,
            indent=4,
        )

    # Save optimal thresholds for step 3
    np.save(f"{output_dir}/features/optimal_thresholds.npy", optimal_thresholds)

    # Evaluate analytical baselines
    tqdm.write("  Evaluating analytical baselines...")
    feat_dir = f"{output_dir}/features"
    distribution = load_feature_group(f"{feat_dir}/distribution.json")
    # Reconstruct V_max mean/quantiles for baselines from distribution features
    # We need the raw V_max for quantile baselines — recompute from the model
    from spiking.metrics import compute_vmax_dataset

    V_max_train, _, _ = compute_vmax_dataset(layer, train_loader)

    baselines = evaluate_all_baselines(optimizer, V_max_train, trained_thresholds)
    with open(f"{output_dir}/baselines.json", "w") as f:
        json.dump(baselines, f, indent=4)

    return {
        "t_objective": t_obj,
        "seed": seed,
        "baseline_accuracy": opt_result["baseline_accuracy"],
        "combined_accuracy": opt_result["combined_accuracy"],
        "conservative_accuracy": opt_result["conservative_accuracy"],
        "avg_per_neuron_improvement": opt_result["avg_per_neuron_improvement"],
        "n_improved": opt_result["n_improved"],
        "n_changed": opt_result["n_changed"],
        "best_baseline": max(baselines, key=lambda b: b["accuracy"]),
    }


def run(
    dataset: str,
    *,
    force: bool = False,
    seeds: list[int] | None = None,
):
    train_loader, val_loader = create_dataset(dataset)

    base_dir = f"logs/{dataset}/threshold_prediction"
    models = find_trained_models(base_dir)
    if seeds:
        models = [(p, t, s) for p, t, s in models if s in seeds]
    if not models:
        print(f"No trained models found under {base_dir}. Run step 1 first.")
        return

    all_results = []
    pbar = tqdm(models, desc="Step 2: Optimize thresholds")
    for model_path, t_obj, seed in pbar:
        output_dir = os.path.dirname(model_path)
        pbar.set_postfix_str(f"tobj={t_obj} seed={seed}")

        if not force and os.path.exists(f"{output_dir}/optimization_results.json"):
            tqdm.write(f"  skip tobj={t_obj} seed={seed} (already complete)")
            continue

        result = optimize_single(
            model_path, t_obj, seed, train_loader, val_loader, output_dir
        )
        all_results.append(result)

    pbar.close()

    if all_results:
        with open(f"{base_dir}/optimization_summary.json", "w") as f:
            json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2: Find per-neuron optimal thresholds and evaluate baselines"
    )
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument(
        "--force", action="store_true", help="Re-run even if results exist"
    )
    parser.add_argument("--seeds", type=int, nargs="+", help="Only run these seeds")
    args = parser.parse_args()
    run(args.dataset, force=args.force, seeds=args.seeds)
