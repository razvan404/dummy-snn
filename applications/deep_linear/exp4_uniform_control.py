import argparse
from pathlib import Path

from applications.datasets import create_dataset
from applications.deep_linear.uniform_thresholds import uniform_thresholds
from applications.deep_linear.sweep_post_training import (
    find_trained_models,
    sweep_post_training,
)

SEED_START = 300
DEFAULT_NUM_SEEDS = 3


def run(dataset: str, *, force: bool = False, num_seeds: int = DEFAULT_NUM_SEEDS):
    seeds = list(range(SEED_START, SEED_START + num_seeds))
    base_dir = Path(f"logs/{dataset}/layer_1")
    model_paths = find_trained_models(base_dir)

    train_loader, val_loader = create_dataset(dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    def fn(model_path, output_dir, seed):
        uniform_thresholds(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=spike_shape,
            seed=seed,
            output_dir=output_dir,
            half_width=0.1,
        )

    sweep_post_training(
        model_paths=model_paths,
        seeds=seeds,
        fn=fn,
        result_subdir="uniform_thresh",
        description="Exp 5: uniform threshold control",
        force=force,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 5: uniform threshold control")
    parser.add_argument("dataset", type=str)
    parser.add_argument(
        "--force", action="store_true", help="re-run even if results exist"
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_NUM_SEEDS)
    args = parser.parse_args()

    run(args.dataset, force=args.force, num_seeds=args.seeds)
