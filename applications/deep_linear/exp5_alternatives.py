import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from applications.common import merge_seed_results
from applications.datasets import create_dataset
from applications.deep_linear.apply_pbtr import apply_pbtr
from applications.deep_linear.activity_threshold import activity_threshold
from applications.deep_linear.weight_threshold import weight_threshold

SEED_START = 100
DEFAULT_NUM_SEEDS = 3

# Each variant: (subdirectory_name, callable, extra_kwargs)
PBTR_VARIANTS = [
    ("pbtr_tau01", dict(tau=0.1, homeostatic=False, sign_only=False)),
    ("pbtr_tau01_homeo", dict(tau=0.1, homeostatic=True, sign_only=False)),
    ("pbtr_tau20_homeo", dict(tau=20.0, homeostatic=True, sign_only=False)),
    ("pbtr_tau01_homeo_sign", dict(tau=0.1, homeostatic=True, sign_only=True)),
]


def _run_pbtr_variant(
    model_path,
    seed_dir,
    variant_name,
    variant_kwargs,
    *,
    train_loader,
    val_loader,
    spike_shape,
    seeds,
    num_epochs,
    force,
    pbar,
    t_target,
):
    variant_dir = seed_dir / variant_name
    for seed in seeds:
        output_dir = str(variant_dir / f"seed_{seed}")
        if not force and os.path.exists(f"{output_dir}/metrics.json"):
            tqdm.write(f"  skip {seed_dir.name}/{variant_name} seed={seed}")
            pbar.update(1)
            continue
        label = f"{seed_dir.name}/{variant_name} seed={seed}"
        pbar.set_postfix_str(label)
        epoch = [1]
        apply_pbtr(
            model_path=str(model_path),
            dataset_loaders=(train_loader, val_loader),
            spike_shape=spike_shape,
            seed=seed,
            output_dir=output_dir,
            num_epochs=num_epochs,
            t_target=t_target,
            on_epoch_end=lambda e, _total: epoch.__setitem__(0, e + 1),
            **variant_kwargs,
        )
        pbar.update(1)
    if any((variant_dir / f"seed_{s}" / "metrics.json").exists() for s in seeds):
        merge_seed_results(str(variant_dir))


def _run_oneshot(
    func,
    model_path,
    seed_dir,
    variant_name,
    *,
    train_loader,
    val_loader,
    spike_shape,
    seeds,
    force,
    pbar,
    t_target,
):
    variant_dir = seed_dir / variant_name
    for seed in seeds:
        output_dir = str(variant_dir / f"seed_{seed}")
        if not force and os.path.exists(f"{output_dir}/metrics.json"):
            tqdm.write(f"  skip {seed_dir.name}/{variant_name} seed={seed}")
            pbar.update(1)
            continue
        pbar.set_postfix_str(f"{seed_dir.name}/{variant_name} seed={seed}")
        func(
            model_path=str(model_path),
            dataset_loaders=(train_loader, val_loader),
            spike_shape=spike_shape,
            seed=seed,
            output_dir=output_dir,
            t_target=t_target,
        )
        pbar.update(1)
    if any((variant_dir / f"seed_{s}" / "metrics.json").exists() for s in seeds):
        merge_seed_results(str(variant_dir))


def run(
    dataset: str,
    *,
    num_epochs: int = 10,
    force: bool = False,
    num_seeds: int = DEFAULT_NUM_SEEDS,
):
    seeds = list(range(SEED_START, SEED_START + num_seeds))
    base_dir = Path(f"logs/{dataset}/layer_1")
    model_paths = [
        p
        for p in sorted(base_dir.rglob("model.pth"))
        if not any(
            part.startswith(
                ("pbtr", "random_thresh", "activity_thresh", "weight_thresh")
            )
            for part in p.parts
        )
    ]
    if not model_paths:
        print(f"No models found under {base_dir}/")
        sys.exit(0)

    train_loader, val_loader = create_dataset(dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    num_variants = len(PBTR_VARIANTS) + 2  # +2 for activity and weight
    total = len(model_paths) * len(seeds) * num_variants
    with tqdm(total=total, desc="Exp 5: alternative post-training") as pbar:
        for model_path in model_paths:
            seed_dir = model_path.parent
            t_target = None
            setup_path = seed_dir / "setup.json"
            if setup_path.exists():
                with open(setup_path) as f:
                    t_target = json.load(f).get("t_objective")

            common = dict(
                train_loader=train_loader,
                val_loader=val_loader,
                spike_shape=spike_shape,
                seeds=seeds,
                force=force,
                pbar=pbar,
                t_target=t_target,
            )

            for variant_name, variant_kwargs in PBTR_VARIANTS:
                _run_pbtr_variant(
                    model_path,
                    seed_dir,
                    variant_name,
                    variant_kwargs,
                    num_epochs=num_epochs,
                    **common,
                )

            _run_oneshot(
                activity_threshold,
                model_path,
                seed_dir,
                "activity_thresh",
                **common,
            )
            _run_oneshot(
                weight_threshold,
                model_path,
                seed_dir,
                "weight_thresh",
                **common,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 5: alternative post-training threshold methods"
    )
    parser.add_argument("dataset", type=str)
    parser.add_argument("--pbtr-epochs", type=int, default=10)
    parser.add_argument(
        "--force", action="store_true", help="re-run even if results exist"
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_NUM_SEEDS)
    args = parser.parse_args()

    run(
        args.dataset,
        num_epochs=args.pbtr_epochs,
        force=args.force,
        num_seeds=args.seeds,
    )
