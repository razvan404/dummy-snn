import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from applications.common import merge_seed_results
from applications.datasets import create_dataset
from applications.deep_linear.apply_pbtr import apply_pbtr

SEEDS = [100, 101, 102]


def run(dataset: str, *, num_epochs: int = 10):
    base_dir = Path(f"logs/{dataset}/layer_1")
    model_paths = [
        p
        for p in sorted(base_dir.rglob("model.pth"))
        if "pbtr" not in p.parts and "random_thresh" not in p.parts
    ]
    if not model_paths:
        print(f"No models found under {base_dir}/")
        sys.exit(0)

    train_loader, val_loader = create_dataset(dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    train_steps = len(train_loader)
    steps = {"train": train_steps}

    total = len(model_paths) * len(SEEDS)
    with tqdm(total=total, desc="Exp 3: PBTR post-training") as pbar:
        for model_path in model_paths:
            seed_dir = model_path.parent
            pbtr_dir = seed_dir / "pbtr"
            for seed in SEEDS:
                label = f"{seed_dir.name} seed={seed}"
                pbar.set_postfix_str(label)
                apply_pbtr(
                    model_path=str(model_path),
                    dataset_loaders=(train_loader, val_loader),
                    spike_shape=spike_shape,
                    seed=seed,
                    output_dir=str(pbtr_dir / f"seed_{seed}"),
                    num_epochs=num_epochs,
                    on_batch_end=lambda idx, _dw, split: (
                        pbar.set_postfix_str(
                            f"{label} {split} {idx+1}/{steps.get(split, '?')}"
                        )
                        if idx % 200 == 0
                        else None
                    ),
                )
                pbar.update(1)
            merge_seed_results(str(pbtr_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 3: PBTR post-training")
    parser.add_argument("dataset", type=str)
    parser.add_argument("--pbtr-epochs", type=int, default=10)
    args = parser.parse_args()

    run(args.dataset, num_epochs=args.pbtr_epochs)
