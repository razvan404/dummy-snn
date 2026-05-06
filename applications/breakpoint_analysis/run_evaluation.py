"""CLI: Evaluate accuracy at breakpoint-derived threshold candidates.

Zooms into the [0.75θ, 1.25θ] range per filter and evaluates accuracy at
actual transition points where spike behavior changes.

Usage:
    python -m applications.breakpoint_analysis.run_evaluation \
        --dataset cifar10 --device cuda --num-candidates 50
"""

import argparse
import json
import logging
import os

from applications.common import create_dataloaders, resolve_params
from applications.breakpoint_analysis.evaluate_breakpoints import (
    compute_breakpoint_candidates,
    evaluate_breakpoint_candidates,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate accuracy at breakpoint-derived thresholds"
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist", choices=["mnist", "cifar10"]
    )
    parser.add_argument("--num-filters", type=int, default=None)
    parser.add_argument("--t-obj", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=50,
        help="Number of breakpoint candidates per filter",
    )
    parser.add_argument("--low-ratio", type=float, default=0.75)
    parser.add_argument("--high-ratio", type=float, default=1.25)
    parser.add_argument(
        "--bp-max-samples",
        type=int,
        default=None,
        help="Max samples for breakpoint computation (None = full dataset)",
    )
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _, _, model_dir = resolve_params(args)
    model_path = f"{model_dir}/model.pth"

    if not os.path.exists(model_path):
        logger.error("No model at %s — run train.py first", model_dir)
        return

    output_path = f"{model_dir}/breakpoint_evaluation.json"
    if not args.force and os.path.exists(output_path):
        logger.info("Exists: %s (use --force)", output_path)
        return

    train_loader, val_loader = create_dataloaders(args.dataset)

    # Check for precomputed breakpoints
    bp_path = f"{model_dir}/breakpoints.npz"
    if not os.path.exists(bp_path):
        bp_path = None
        logger.info("No precomputed breakpoints — will compute from scratch")

    # Load setup for t_target
    setup_path = f"{model_dir}/setup.json"
    t_target = None
    if os.path.exists(setup_path):
        with open(setup_path) as f:
            setup = json.load(f)
        t_target = setup.get("target_timestamp")

    # Step 1: Get breakpoint candidates per filter
    candidates, thresholds = compute_breakpoint_candidates(
        model_path=model_path,
        loader=train_loader,
        num_candidates=args.num_candidates,
        low_ratio=args.low_ratio,
        high_ratio=args.high_ratio,
        device=args.device,
        chunk_size=args.chunk_size,
        max_samples=args.bp_max_samples,
        breakpoints_path=bp_path,
    )

    n_cands = [len(c) for c in candidates]
    logger.info(
        "Candidates per filter: min=%d, median=%d, max=%d",
        min(n_cands),
        int(sorted(n_cands)[len(n_cands) // 2]),
        max(n_cands),
    )

    # Step 2: Evaluate each candidate
    results = evaluate_breakpoint_candidates(
        model_path=model_path,
        train_loader=train_loader,
        val_loader=val_loader,
        candidates_per_filter=candidates,
        t_target=t_target,
        pool_size=args.pool_size,
        device=args.device,
        chunk_size=args.chunk_size,
        alpha=args.alpha,
    )

    # Print summary
    baseline = results["baseline_accuracy"]
    improved = 0
    total_gain = 0.0
    for fr in results["filters"]:
        delta = fr["best_val_accuracy"] - baseline
        if delta > 0:
            improved += 1
            total_gain += delta

    logger.info(
        "Baseline: %.4f | %d/%d filters have individually better thresholds | "
        "avg gain when better: %.4f",
        baseline,
        improved,
        results["num_filters"],
        total_gain / max(improved, 1),
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
