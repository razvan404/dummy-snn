import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm

from applications.common import merge_seed_results
from applications.datasets import create_dataset
from applications.deep_linear.random_thresholds import random_thresholds

SEED_START = 200
DEFAULT_NUM_SEEDS = 5


def run(dataset: str, *, force: bool = False, num_seeds: int = DEFAULT_NUM_SEEDS):
    seeds = list(range(SEED_START, SEED_START + num_seeds))
    base_dir = Path(f"logs/{dataset}/layer_1")
    model_paths = [
        p
        for p in sorted(base_dir.rglob("model.pth"))
        if not any(part.startswith("pbtr") for part in p.parts)
        and "random_thresh" not in p.parts
    ]
    if not model_paths:
        print(f"No models found under {base_dir}/")
        sys.exit(0)

    train_loader, val_loader = create_dataset(dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    total = len(model_paths) * len(seeds)
    with tqdm(total=total, desc="Exp 4: random threshold control") as pbar:
        for model_path in model_paths:
            seed_dir = model_path.parent
            ctrl_dir = seed_dir / "random_thresh"
            for seed in seeds:
                output_dir = str(ctrl_dir / f"seed_{seed}")
                if not force and os.path.exists(f"{output_dir}/metrics.json"):
                    tqdm.write(f"  skip {seed_dir.name} seed={seed} (already complete)")
                    pbar.update(1)
                    continue
                pbar.set_postfix_str(f"{seed_dir.name} seed={seed}")
                random_thresholds(
                    model_path=str(model_path),
                    dataset_loaders=(train_loader, val_loader),
                    spike_shape=spike_shape,
                    seed=seed,
                    output_dir=output_dir,
                )
                pbar.update(1)
            merge_seed_results(str(ctrl_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 4: random threshold control")
    parser.add_argument("dataset", type=str)
    parser.add_argument("--force", action="store_true", help="re-run even if results exist")
    parser.add_argument("--seeds", type=int, default=DEFAULT_NUM_SEEDS)
    args = parser.parse_args()

    run(args.dataset, force=args.force, num_seeds=args.seeds)
