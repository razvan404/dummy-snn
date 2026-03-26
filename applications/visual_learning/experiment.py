import json
import logging
import os

from applications.common import aggregate_metrics
from applications.visual_learning.patch_train import train_model

logger = logging.getLogger(__name__)


SEEDS = [42, 123, 456]
STDP_VARIANTS = ["multiplicative", "biological"]


def run_experiment(
    num_filters: int = 256,
    num_epochs: int = 100,
    processed_dir: str = "data/processed-cifar10",
    output_dir: str = "logs/visual_learning_experiment",
    seeds: list[int] | None = None,
    variants: list[str] | None = None,
):
    """Run the full comparison experiment.

    For each STDP variant, trains with multiple seeds and computes
    mean ± std accuracy. Saves per-variant summaries.
    """
    if seeds is None:
        seeds = SEEDS
    if variants is None:
        variants = STDP_VARIANTS

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for variant in variants:
        logger.info("\n%s\nSTDP variant: %s\n%s", "=" * 60, variant, "=" * 60)

        variant_metrics = []
        for seed in seeds:
            run_dir = f"{output_dir}/{variant}/seed_{seed}"
            metrics = train_model(
                seed=seed,
                stdp_variant=variant,
                num_filters=num_filters,
                num_epochs=num_epochs,
                processed_dir=processed_dir,
                output_dir=run_dir,
            )
            variant_metrics.append(metrics)

        summary = aggregate_metrics(variant_metrics)
        results[variant] = summary

        val_acc = summary["validation"]["accuracy"]
        logger.info(
            "%s STDP — test accuracy: %.4f +/- %.4f",
            variant,
            val_acc["mean"],
            val_acc["std"],
        )

    # Save combined results
    with open(f"{output_dir}/comparison.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info("\n%s\nComparison summary:", "=" * 60)
    for variant, summary in results.items():
        val_acc = summary["validation"]["accuracy"]
        logger.info("  %15s: %.4f +/- %.4f", variant, val_acc["mean"], val_acc["std"])
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Falez 2020 experiment sweep")
    parser.add_argument("--num-filters", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument(
        "--output-dir", type=str, default="logs/visual_learning_experiment"
    )
    args = parser.parse_args()

    run_experiment(
        num_filters=args.num_filters,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
    )
