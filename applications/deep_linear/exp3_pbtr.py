import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from applications.common import merge_seed_results
from applications.datasets import create_dataset
from applications.deep_linear.apply_pbtr import apply_pbtr

SEED_START = 100
DEFAULT_NUM_SEEDS = 3


def run(
    dataset: str,
    *,
    num_epochs: int = 10,
    force: bool = False,
    num_seeds: int = DEFAULT_NUM_SEEDS,
    sign_only: bool = False,
):
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

    train_steps = len(train_loader)
    steps = {"train": train_steps}

    total = len(model_paths) * len(seeds)
    with tqdm(total=total, desc="Exp 3: PBTR post-training") as pbar:
        for model_path in model_paths:
            seed_dir = model_path.parent
            setup_path = seed_dir / "setup.json"
            t_target = None
            if setup_path.exists():
                with open(setup_path) as f:
                    t_target = json.load(f).get("t_objective")
            other_dir = seed_dir / ("pbtr" if sign_only else "pbtr_sign")
            if not force and other_dir.exists():
                tqdm.write(f"  skip {seed_dir.name} ({other_dir.name}/ exists, flag mismatch)")
                pbar.update(len(seeds))
                continue
            pbtr_dir = seed_dir / ("pbtr_sign" if sign_only else "pbtr")
            for seed in seeds:
                output_dir = str(pbtr_dir / f"seed_{seed}")
                if not force and os.path.exists(f"{output_dir}/metrics.json"):
                    tqdm.write(f"  skip {seed_dir.name} seed={seed} (already complete)")
                    pbar.update(1)
                    continue
                label = f"{seed_dir.name} seed={seed}"
                pbar.set_postfix_str(label)
                apply_pbtr(
                    model_path=str(model_path),
                    dataset_loaders=(train_loader, val_loader),
                    spike_shape=spike_shape,
                    seed=seed,
                    output_dir=output_dir,
                    num_epochs=num_epochs,
                    t_target=t_target,
                    sign_only=sign_only,
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
    parser.add_argument(
        "--force", action="store_true", help="re-run even if results exist"
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--sign-only", action="store_true")
    args = parser.parse_args()

    run(
        args.dataset,
        num_epochs=args.pbtr_epochs,
        force=args.force,
        num_seeds=args.seeds,
        sign_only=args.sign_only,
    )
