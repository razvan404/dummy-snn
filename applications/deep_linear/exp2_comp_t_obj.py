import argparse

from tqdm import tqdm

from applications.common import merge_seed_results
from applications.datasets import create_dataset
from applications.deep_linear.train import train_layer

DATASET_THRESHOLDS = {
    "mnist": 156.8,
    "mnist_subset": 156.8,
    "fashion_mnist": 156.8,
    "cifar10": 204.8,
}

SEEDS = [1, 2, 3, 4, 5]
T_OBJECTIVES = [round(0.4 + v * 0.05, 2) for v in range(12)]  # 0.4 to 0.95


def run(dataset: str, *, num_epochs: int = 30):
    thresh = DATASET_THRESHOLDS[dataset]

    train_loader, val_loader = create_dataset(dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    train_steps = len(train_loader)
    val_steps = len(val_loader)
    steps = {"train": train_steps, "val": val_steps}

    total = len(T_OBJECTIVES) * len(SEEDS)
    with tqdm(total=total, desc="Exp 2: t_objective sweep") as pbar:
        for t_obj in T_OBJECTIVES:
            base_dir = f"logs/{dataset}/layer_1/comp_t_obj/tobj_{t_obj}"
            for seed in SEEDS:
                pbar.set_postfix_str(f"t_obj={t_obj} seed={seed}")
                train_layer(
                    dataset_loaders=(train_loader, val_loader),
                    spike_shape=spike_shape,
                    seed=seed,
                    avg_threshold=thresh,
                    output_dir=f"{base_dir}/seed_{seed}",
                    num_epochs=num_epochs,
                    t_objective=t_obj,
                    on_batch_end=lambda idx, _dw, split: (
                        pbar.set_postfix_str(
                            f"t_obj={t_obj} seed={seed} {split} {idx+1}/{steps[split]}"
                        )
                        if idx % 200 == 0
                        else None
                    ),
                )
                pbar.update(1)
            merge_seed_results(base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 2: competitive + target-timestamp sweep"
    )
    parser.add_argument("dataset", choices=list(DATASET_THRESHOLDS))
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    run(args.dataset, num_epochs=args.epochs)
