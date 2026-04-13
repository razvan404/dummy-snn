"""Iterative coordinate descent threshold optimization for a single ordering."""

import argparse
import json
import logging
import os

from applications.common import create_dataloaders, resolve_params
from applications.threshold_research.filter_ordering import ORDERINGS
from applications.threshold_research.iterative_optimization import (
    iterative_coordinate_descent,
    plot_convergence,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize conv SNN thresholds with iterative coordinate descent"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["mnist", "cifar10"],
    )
    parser.add_argument("--num-filters", type=int, default=None)
    parser.add_argument("--t-obj", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--num-rounds", type=int, default=25)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument(
        "--ordering",
        type=str,
        choices=ORDERINGS,
        default="descending_importance",
        help="Filter ordering strategy",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _, _, model_dir = resolve_params(args)

    if not os.path.exists(f"{model_dir}/model.pth"):
        logger.error("No model at %s — run train.py first", model_dir)
        return

    output_path = f"{model_dir}/iterative_optimization_{args.ordering}.json"
    plot_path = f"{model_dir}/iterative_optimization_{args.ordering}_convergence.png"

    if not args.force and os.path.exists(output_path):
        logger.info("Exists: %s (use --force)", output_path)
        return

    with open(f"{model_dir}/setup.json") as f:
        setup = json.load(f)

    train_loader, val_loader = create_dataloaders(args.dataset)

    results = iterative_coordinate_descent(
        model_path=f"{model_dir}/model.pth",
        dataset_loaders=(train_loader, val_loader),
        t_target=setup.get("target_timestamp"),
        pool_size=setup.get("pool_size", 2),
        min_threshold=setup.get("min_threshold", 1.0),
        device=args.device,
        chunk_size=args.chunk_size,
        num_rounds=args.num_rounds,
        step_size=args.step_size,
        ordering=args.ordering,
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    plot_convergence(results, plot_path)

    logger.info(
        "[%s] train %.4f->%.4f (%+.4f), val %.4f->%.4f (%+.4f), %d rounds",
        args.ordering,
        results["baseline_train_accuracy"],
        results["final_train_accuracy"],
        results["final_train_accuracy"] - results["baseline_train_accuracy"],
        results["baseline_val_accuracy"],
        results["final_val_accuracy"],
        results["final_val_accuracy"] - results["baseline_val_accuracy"],
        len(results["rounds"]),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
