"""CLI: Analyze threshold breakpoints for a trained conv SNN model.

Computes the exact set of cumulative potential values (breakpoints) per filter
across the dataset. Between consecutive breakpoints, the accuracy landscape is
flat — the spike times don't change. This tells us whether exact piecewise
optimization is feasible (few breakpoints) or requires subsampling (many).

Usage:
    python -m applications.breakpoint_analysis.analyze --dataset mnist --device cuda
    python -m applications.breakpoint_analysis.analyze --dataset cifar10 --max-samples 5000
"""

import argparse
import logging
import os

import numpy as np

from applications.common import create_dataloaders, resolve_params
from applications.breakpoint_analysis.compute_breakpoints import (
    compute_breakpoints,
    print_breakpoint_summary,
    save_breakpoint_results,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze per-filter threshold breakpoints"
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist", choices=["mnist", "cifar10"]
    )
    parser.add_argument("--num-filters", type=int, default=None)
    parser.add_argument("--t-obj", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap on samples to process (default: full dataset)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Which data split to analyze",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _, _, model_dir = resolve_params(args)
    model_path = f"{model_dir}/model.pth"

    if not os.path.exists(model_path):
        logger.error("No model at %s — run train.py first", model_dir)
        return

    output_path = f"{model_dir}/breakpoint_analysis.json"
    if not args.force and os.path.exists(output_path):
        logger.info("Exists: %s (use --force)", output_path)
        return

    train_loader, val_loader = create_dataloaders(args.dataset)
    loader = train_loader if args.split == "train" else val_loader

    stats = compute_breakpoints(
        model_path=model_path,
        loader=loader,
        device=args.device,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
    )

    print_breakpoint_summary(stats)

    # Save raw breakpoints as .npz for downstream optimization
    bp_dict = {f"filter_{s.filter_idx}": s.breakpoints for s in stats}
    bp_dict["thresholds"] = np.array([s.current_threshold for s in stats])
    npz_path = f"{model_dir}/breakpoints.npz"
    np.savez_compressed(npz_path, **bp_dict)
    logger.info("Saved raw breakpoints to %s", npz_path)

    save_breakpoint_results(stats, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
