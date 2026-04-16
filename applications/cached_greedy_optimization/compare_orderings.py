"""Compare greedy optimization results across orderings with Ridge + SVC."""

import argparse
import json
import logging
import os

import numpy as np
import torch

from applications.cached_greedy_optimization.optimize import build_features_from_levels
from spiking.evaluation.eval_classifier import evaluate_classifier
from spiking.evaluation.ridge_column_swap import RidgeColumnSwap

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare greedy orderings")
    parser.add_argument("--cache-path", required=True)
    parser.add_argument(
        "--result-dirs",
        nargs="+",
        required=True,
        help="Directories containing results.json from each ordering run",
    )
    args = parser.parse_args()

    # Load cache
    logger.info("Loading cache...")
    cache = torch.load(args.cache_path, weights_only=False)
    train_cache = cache["train_cache"]
    test_cache = cache["test_cache"]
    y_train = cache["y_train"]
    y_test = cache["y_test"]
    fractions = cache["perturbation_fractions"]
    pool_dim = cache["pool_size"] ** 2
    num_filters = train_cache.shape[0]
    zero_idx = fractions.index(0.0)

    # Baseline features
    baseline_levels = np.full(num_filters, zero_idx, dtype=int)
    X_train_base = build_features_from_levels(train_cache, baseline_levels, pool_dim)
    X_test_base = build_features_from_levels(test_cache, baseline_levels, pool_dim)

    # Evaluate baseline
    logger.info("=== Baseline ===")
    ridge = RidgeColumnSwap(alpha=1.0)
    ridge_train, ridge_val = evaluate_classifier(
        X_train_base, y_train, X_test_base, y_test, classifier=ridge
    )
    svc_train, svc_val = evaluate_classifier(X_train_base, y_train, X_test_base, y_test)
    logger.info(
        "  Ridge — train: %.4f, val: %.4f",
        ridge_train["accuracy"],
        ridge_val["accuracy"],
    )
    logger.info(
        "  SVC   — train: %.4f, val: %.4f",
        svc_train["accuracy"],
        svc_val["accuracy"],
    )

    results_table = []
    results_table.append(
        {
            "ordering": "baseline",
            "ridge_train": ridge_train["accuracy"],
            "ridge_val": ridge_val["accuracy"],
            "svc_train": svc_train["accuracy"],
            "svc_val": svc_val["accuracy"],
        }
    )

    # Evaluate each ordering
    for result_dir in args.result_dirs:
        results_path = os.path.join(result_dir, "results.json")
        if not os.path.exists(results_path):
            logger.warning("Skipping %s — no results.json", result_dir)
            continue

        with open(results_path) as f:
            results = json.load(f)

        ordering = results["config"]["ordering"]
        seed = results["config"].get("seed", 1)
        levels = np.array(results["current_levels"], dtype=int)
        n_changed = int((levels != zero_idx).sum())

        X_train = build_features_from_levels(train_cache, levels, pool_dim)
        X_test = build_features_from_levels(test_cache, levels, pool_dim)

        ridge = RidgeColumnSwap(alpha=1.0)
        ridge_train, ridge_val = evaluate_classifier(
            X_train, y_train, X_test, y_test, classifier=ridge
        )
        svc_train, svc_val = evaluate_classifier(X_train, y_train, X_test, y_test)

        label = f"{ordering}" + (f" (s{seed})" if ordering == "random" else "")
        logger.info("=== %s (%d neurons changed) ===", label, n_changed)
        logger.info(
            "  Ridge — train: %.4f, val: %.4f",
            ridge_train["accuracy"],
            ridge_val["accuracy"],
        )
        logger.info(
            "  SVC   — train: %.4f, val: %.4f",
            svc_train["accuracy"],
            svc_val["accuracy"],
        )

        results_table.append(
            {
                "ordering": label,
                "n_changed": n_changed,
                "ridge_train": ridge_train["accuracy"],
                "ridge_val": ridge_val["accuracy"],
                "svc_train": svc_train["accuracy"],
                "svc_val": svc_val["accuracy"],
            }
        )

    # Summary table
    logger.info("")
    logger.info(
        "%-25s | %s | %s | %s | %s",
        "Ordering",
        "Ridge Train",
        "Ridge Val",
        "SVC Train",
        "SVC Val",
    )
    logger.info("-" * 85)
    for row in results_table:
        logger.info(
            "%-25s | %.4f      | %.4f    | %.4f    | %.4f",
            row["ordering"],
            row["ridge_train"],
            row["ridge_val"],
            row["svc_train"],
            row["svc_val"],
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
