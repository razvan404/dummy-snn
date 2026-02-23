import copy
import json
import os

import tqdm

from applications.common import set_seed, evaluate_model, aggregate_metrics
from applications.datasets import create_dataset
from applications import default_hyperparams
from applications.deep_linear.model import (
    ARCHITECTURE,
    create_model,
    train_layerwise,
    apply_pba,
)
from spiking.utils import save_model

# ── User edits these between runs ──────────────────────────────────
NUM_LAYERS = 1  # increase: 1 → 2 → 3
NUM_EPOCHS_PER_LAYER = 50
PBA_EPOCHS = 20
PBA_KWARGS = {
    "tau": 20.0,
    "learning_rate": 0.1,
    "min_threshold": 1.0,
    "max_threshold": 100.0,
}
SEEDS = list(range(1, 11))
DATASET = "mnist"

setup = {
    "threshold_init": {**default_hyperparams.THRESHOLD_INIT},
    "threshold_adaptation": {**default_hyperparams.THRESHOLD_ADAPTATION},
    "stdp": {**default_hyperparams.STDP},
}
# ───────────────────────────────────────────────────────────────────


def _save_seed_artifacts(model, metrics, seed_dir):
    os.makedirs(seed_dir, exist_ok=True)
    save_model(model, f"{seed_dir}/model.pth")
    with open(f"{seed_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


def _save_condition_summary(all_metrics, condition_dir):
    summary = aggregate_metrics(all_metrics)
    with open(f"{condition_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    with open(f"{condition_dir}/all_seeds_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
    return summary


def main():
    train_loader, val_loader = create_dataset(DATASET)
    image_shape = train_loader.dataset.image_shape
    spike_shape = (2, *image_shape)
    num_inputs = 2 * image_shape[0] * image_shape[1]

    base_dir = f"logs/linear-deep/benchmark/{NUM_LAYERS}_layers"
    no_pba_dir = f"{base_dir}/no_pba"
    with_pba_dir = f"{base_dir}/with_pba"
    os.makedirs(no_pba_dir, exist_ok=True)
    os.makedirs(with_pba_dir, exist_ok=True)

    no_pba_metrics = []
    with_pba_metrics = []

    for seed in tqdm.tqdm(SEEDS, desc="Seeds"):
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        set_seed(seed)
        model = create_model(setup, num_inputs, ARCHITECTURE)
        train_layerwise(
            model,
            setup,
            train_loader,
            val_loader,
            spike_shape,
            num_layers=NUM_LAYERS,
            num_epochs_per_layer=NUM_EPOCHS_PER_LAYER,
        )

        # Baseline (no PBA)
        model_no_pba = copy.deepcopy(model)
        train_m, val_m = evaluate_model(
            model_no_pba,
            train_loader,
            val_loader,
            spike_shape,
        )
        metrics_no_pba = {"train": train_m, "validation": val_m}
        no_pba_metrics.append(metrics_no_pba)
        _save_seed_artifacts(model_no_pba, metrics_no_pba, f"{no_pba_dir}/seed_{seed}")
        print(f"  no PBA  — val accuracy: {val_m['accuracy']:.4f}")

        # With PBA
        model_pba = copy.deepcopy(model)
        apply_pba(
            model_pba,
            train_loader,
            spike_shape,
            num_layers=NUM_LAYERS,
            pba_kwargs=PBA_KWARGS,
            num_epochs=PBA_EPOCHS,
        )
        train_m, val_m = evaluate_model(
            model_pba,
            train_loader,
            val_loader,
            spike_shape,
        )
        metrics_pba = {"train": train_m, "validation": val_m}
        with_pba_metrics.append(metrics_pba)
        _save_seed_artifacts(model_pba, metrics_pba, f"{with_pba_dir}/seed_{seed}")
        print(f"  with PBA — val accuracy: {val_m['accuracy']:.4f}")

    # Save summaries
    no_pba_summary = _save_condition_summary(no_pba_metrics, no_pba_dir)
    pba_summary = _save_condition_summary(with_pba_metrics, with_pba_dir)

    # Print comparison
    print(f"\n{'='*60}")
    print(f"Comparison ({NUM_LAYERS} layers, {len(SEEDS)} seeds)")
    print(f"{'='*60}")
    for condition, summary in [("no PBA", no_pba_summary), ("with PBA", pba_summary)]:
        val_acc = summary["validation"]["accuracy"]
        print(f"  {condition:10s}: {val_acc['mean']:.4f} ± {val_acc['std']:.4f}")


if __name__ == "__main__":
    main()
