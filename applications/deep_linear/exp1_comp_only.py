import argparse
import math

from tqdm import tqdm

from applications.common import merge_seed_results
from applications.datasets import DATASETS, create_dataset
from applications.deep_linear.train import train_layer
SEEDS = [1, 2, 3, 4, 5]


def run(dataset: str, *, num_epochs: int = 10):
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

    total = len(thresholds) * len(SEEDS)
    with tqdm(total=total, desc="Exp 1: threshold sweep") as pbar:
        for thresh in thresholds:
            base_dir = f"logs/{dataset}/layer_1/comp_only/thresh_{thresh}"
            for seed in SEEDS:
                pbar.set_postfix_str(f"thresh={thresh} seed={seed}")
                epoch = [1]
                train_layer(
                    dataset_loaders=(train_loader, val_loader),
                    spike_shape=spike_shape,
                    seed=seed,
                    avg_threshold=thresh,
                    output_dir=f"{base_dir}/seed_{seed}",
                    num_epochs=num_epochs,
                    on_batch_end=lambda idx, _dw, _split: (
                        pbar.set_postfix_str(
                            f"thresh={thresh} seed={seed} epoch={epoch[0]}/{num_epochs} {idx+1}/{train_steps}"
                        )
                        if idx % 200 == 0
                        else None
                    ),
                    on_epoch_end=lambda e, _total: epoch.__setitem__(0, e + 1),
                )
                pbar.update(1)
            merge_seed_results(base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 1: competitive-only threshold sweep"
    )
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    run(args.dataset, num_epochs=args.epochs)
