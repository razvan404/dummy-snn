import argparse
import logging
import os

from applications.common import merge_seed_results
from applications.conv_learning.train import train_model
from applications.conv_learning.evaluate import evaluate_model_dir

logger = logging.getLogger(__name__)

DEFAULT_T_OBJS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_DATASETS = ["mnist", "cifar10"]


def run_sweep(
    *,
    mode: str = "train",
    datasets: list[str] | None = None,
    t_objs: list[float] | None = None,
    seeds: list[int] | None = None,
    num_filters: int | None = None,
    num_epochs: int | None = None,
    base_dir: str = "logs/sweep",
    device: str = "cpu",
    chunk_size: int = 512,
):
    """Run a t_obj × seed sweep across datasets.

    :param mode: 'train', 'evaluate', or 'both'.
    :param datasets: List of dataset names to sweep.
    :param t_objs: List of target timestamps to sweep.
    :param seeds: List of random seeds.
    :param num_filters: Override filter count (uses paper default if None).
    :param num_epochs: Override epoch count (uses paper default if None).
    :param base_dir: Root output directory.
    :param device: Device for evaluation inference.
    :param chunk_size: Batch size for evaluation inference.
    """
    if datasets is None:
        datasets = DEFAULT_DATASETS
    if t_objs is None:
        t_objs = DEFAULT_T_OBJS
    if seeds is None:
        seeds = DEFAULT_SEEDS

    do_train = mode in ("train", "both")
    do_eval = mode in ("evaluate", "both")

    total = len(datasets) * len(t_objs) * len(seeds)
    done = 0

    for dataset in datasets:
        for t_obj in t_objs:
            tobj_dir = f"{base_dir}/{dataset}/tobj_{t_obj:.2f}"

            for seed in seeds:
                output_dir = f"{tobj_dir}/seed_{seed}"
                done += 1
                tag = f"[{done}/{total}] {dataset} t_obj={t_obj:.2f} seed={seed}"

                if do_train:
                    if os.path.exists(f"{output_dir}/model.pth"):
                        logger.info("%s — model exists, skipping training", tag)
                    else:
                        logger.info("%s — training...", tag)
                        train_model(
                            dataset=dataset,
                            seed=seed,
                            t_obj=t_obj,
                            num_filters=num_filters,
                            num_epochs=num_epochs,
                            output_dir=output_dir,
                        )

                if do_eval:
                    if os.path.exists(f"{output_dir}/metrics.json"):
                        logger.info("%s — metrics exist, skipping evaluation", tag)
                    elif not os.path.exists(f"{output_dir}/model.pth"):
                        logger.warning("%s — no model found, skipping evaluation", tag)
                    else:
                        logger.info("%s — evaluating...", tag)
                        evaluate_model_dir(
                            model_dir=output_dir,
                            device=device,
                            chunk_size=chunk_size,
                        )

            # Aggregate seeds for this t_obj after evaluation
            if do_eval and os.path.isdir(tobj_dir):
                try:
                    merge_seed_results(tobj_dir)
                    logger.info(
                        "Aggregated results for %s t_obj=%.2f", dataset, t_obj
                    )
                except Exception as e:
                    logger.warning("Could not aggregate %s: %s", tobj_dir, e)

    logger.info("Sweep complete.")


def _parse_float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",")]


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Sweep t_obj × seed for conv SNN experiments"
    )
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "evaluate", "both"],
        help="'train' (CPU), 'evaluate' (GPU), or 'both'",
    )
    parser.add_argument(
        "--datasets", type=str, default=None,
        help="Comma-separated dataset names (default: mnist,cifar10)",
    )
    parser.add_argument(
        "--t-objs", type=str, default=None,
        help="Comma-separated t_obj values (default: 0.60,0.65,...,0.95)",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seeds (default: 1,2,3,4,5)",
    )
    parser.add_argument("--num-filters", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--base-dir", type=str, default="logs/sweep")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()

    run_sweep(
        mode=args.mode,
        datasets=args.datasets.split(",") if args.datasets else None,
        t_objs=_parse_float_list(args.t_objs) if args.t_objs else None,
        seeds=_parse_int_list(args.seeds) if args.seeds else None,
        num_filters=args.num_filters,
        num_epochs=args.num_epochs,
        base_dir=args.base_dir,
        device=args.device,
        chunk_size=args.chunk_size,
    )
