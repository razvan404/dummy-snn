import argparse
import math
import os

from tqdm import tqdm

from applications.common import merge_seed_results
from applications.datasets import DATASETS, create_dataset
from applications.deep_linear.progress_callbacks import make_progress_callbacks
from applications.deep_linear.train import train_layer

SEED_START = 1
DEFAULT_NUM_SEEDS = 5


def run(dataset: str, *, num_epochs: int = 10, force: bool = False, num_seeds: int = DEFAULT_NUM_SEEDS):
    seeds = list(range(SEED_START, SEED_START + num_seeds))
    train_loader, val_loader = create_dataset(dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    num_inputs = math.prod(spike_shape)
    low = num_inputs / 40
    high = num_inputs / 10
    step = num_inputs / 160

    thresholds = []
    thresh = low
    while thresh <= high + 1e-9:
        thresholds.append(round(thresh, 1))
        thresh += step

    train_steps = len(train_loader)

    total = len(thresholds) * len(seeds)
    with tqdm(total=total, desc="Exp 1: threshold sweep") as pbar:
        for thresh in thresholds:
            base_dir = f"logs/{dataset}/layer_1/comp_only/thresh_{thresh}"
            for seed in seeds:
                output_dir = f"{base_dir}/seed_{seed}"
                if not force and os.path.exists(f"{output_dir}/metrics.json"):
                    tqdm.write(f"  skip thresh={thresh} seed={seed} (already complete)")
                    pbar.update(1)
                    continue
                label = f"thresh={thresh} seed={seed}"
                pbar.set_postfix_str(label)
                on_batch_end, on_epoch_end = make_progress_callbacks(
                    pbar, label, num_epochs, {"train": train_steps},
                )
                train_layer(
                    dataset_loaders=(train_loader, val_loader),
                    spike_shape=spike_shape,
                    seed=seed,
                    avg_threshold=thresh,
                    output_dir=output_dir,
                    num_epochs=num_epochs,
                    on_batch_end=on_batch_end,
                    on_epoch_end=on_epoch_end,
                )
                pbar.update(1)
            merge_seed_results(base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 1: competitive-only threshold sweep"
    )
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--force", action="store_true", help="re-run even if results exist"
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_NUM_SEEDS)
    args = parser.parse_args()

    run(args.dataset, num_epochs=args.epochs, force=args.force, num_seeds=args.seeds)
