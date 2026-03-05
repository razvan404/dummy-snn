import argparse
import json
import sys
from pathlib import Path

from applications.datasets import create_dataset
from applications.deep_linear.apply_pbtr import apply_pbtr
from applications.deep_linear.activity_threshold import activity_threshold
from applications.deep_linear.percentile_threshold import percentile_threshold
from applications.deep_linear.random_thresholds import random_thresholds
from applications.deep_linear.weight_threshold import weight_threshold

PBTR_VARIANTS = [
    ("pbtr_tau01", dict(tau=0.1, homeostatic=False, sign_only=False)),
    ("pbtr_tau01_homeo", dict(tau=0.1, homeostatic=True, sign_only=False)),
    ("pbtr_tau20_homeo", dict(tau=20.0, homeostatic=True, sign_only=False)),
    ("pbtr_tau01_homeo_sign", dict(tau=0.1, homeostatic=True, sign_only=True)),
]

RANDOM_STDS = [0.1, 1.0, 3.0, 6.0]

SEED = 100


def run(
    model_path: str,
    dataset: str,
    *,
    num_epochs: int = 10,
    force: bool = False,
):
    model_dir = Path(model_path).parent
    train_loader, val_loader = create_dataset(dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    t_target = None
    setup_path = model_dir / "setup.json"
    if setup_path.exists():
        with open(setup_path) as f:
            t_target = json.load(f).get("t_objective")

    # Load baseline
    baseline_path = model_dir / "metrics.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
    else:
        baseline = None

    results = {}

    # PBTR variants
    for name, kwargs in PBTR_VARIANTS:
        output_dir = str(model_dir / name / f"seed_{SEED}")
        metrics_path = f"{output_dir}/metrics.json"
        if not force and Path(metrics_path).exists():
            print(f"  skip {name} (already exists)")
            with open(metrics_path) as f:
                results[name] = json.load(f)
            continue
        print(f"  running {name}...")
        apply_pbtr(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=spike_shape,
            seed=SEED,
            output_dir=output_dir,
            num_epochs=num_epochs,
            t_target=t_target,
            **kwargs,
        )
        with open(metrics_path) as f:
            results[name] = json.load(f)

    # One-shot methods
    for name, func in [
        ("activity_thresh", activity_threshold),
        ("weight_thresh", weight_threshold),
    ]:
        output_dir = str(model_dir / name / f"seed_{SEED}")
        metrics_path = f"{output_dir}/metrics.json"
        if not force and Path(metrics_path).exists():
            print(f"  skip {name} (already exists)")
            with open(metrics_path) as f:
                results[name] = json.load(f)
            continue
        print(f"  running {name}...")
        func(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=spike_shape,
            seed=SEED,
            output_dir=output_dir,
            t_target=t_target,
        )
        with open(metrics_path) as f:
            results[name] = json.load(f)

    # Random controls at multiple std values
    for std_val in RANDOM_STDS:
        name = f"random_std{std_val}"
        output_dir = str(model_dir / name / f"seed_{SEED}")
        metrics_path = f"{output_dir}/metrics.json"
        if not force and Path(metrics_path).exists():
            print(f"  skip {name} (already exists)")
            with open(metrics_path) as f:
                results[name] = json.load(f)
            continue
        print(f"  running {name}...")
        random_thresholds(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=spike_shape,
            seed=SEED,
            output_dir=output_dir,
            std=std_val,
        )
        with open(metrics_path) as f:
            results[name] = json.load(f)

    # Percentile-based threshold calibration
    for pct in [30, 50, 70]:
        name = f"percentile_{pct}"
        output_dir = str(model_dir / name / f"seed_{SEED}")
        metrics_path = f"{output_dir}/metrics.json"
        if not force and Path(metrics_path).exists():
            print(f"  skip {name} (already exists)")
            with open(metrics_path) as f:
                results[name] = json.load(f)
            continue
        print(f"  running {name}...")
        percentile_threshold(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=spike_shape,
            seed=SEED,
            output_dir=output_dir,
            percentile=float(pct),
            t_target=t_target,
        )
        with open(metrics_path) as f:
            results[name] = json.load(f)

    # Print comparison table
    print()
    print(f"Model: {model_path}")
    print(f"{'Method':<25} {'Train Acc':>10} {'Val Acc':>10} {'Delta':>8}")
    print("-" * 55)

    if baseline:
        base_val = baseline["validation"]["accuracy"] * 100
        print(
            f"{'baseline':<25} {baseline['train']['accuracy']*100:>9.2f}% {base_val:>9.2f}%"
        )
    else:
        base_val = None

    all_methods = (
        [v[0] for v in PBTR_VARIANTS]
        + ["activity_thresh", "weight_thresh"]
        + [f"random_std{s}" for s in RANDOM_STDS]
        + [f"percentile_{p}" for p in [30, 50, 70]]
    )
    for name in all_methods:
        if name not in results:
            continue
        m = results[name]
        val_acc = m["validation"]["accuracy"] * 100
        train_acc = m["train"]["accuracy"] * 100
        delta = f"{val_acc - base_val:+.2f}%" if base_val else ""
        print(f"{name:<25} {train_acc:>9.2f}% {val_acc:>9.2f}% {delta:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick single-seed evaluation of post-training methods"
    )
    parser.add_argument("model_path", type=str, nargs="?", default=None)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--pbtr-epochs", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.model_path is None:
        # Default: first params_establish model
        base = Path(f"logs/{args.dataset}/layer_1/params_establish")
        candidates = sorted(base.glob("seed_*/model.pth"))
        if not candidates:
            print(f"No models found under {base}/")
            sys.exit(1)
        args.model_path = str(candidates[0])
        print(f"Using {args.model_path}")

    run(
        args.model_path,
        args.dataset,
        num_epochs=args.pbtr_epochs,
        force=args.force,
    )
