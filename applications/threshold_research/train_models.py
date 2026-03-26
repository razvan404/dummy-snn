import argparse
import json
import math
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from applications.common import merge_seed_results, set_seed, evaluate_model
from applications.datasets import DATASETS, create_dataset
from applications.deep_linear.model import create_model
from applications.deep_linear.progress_callbacks import make_progress_callbacks
from applications.deep_linear.training_plots import (
    save_winner_counts,
    save_threshold_distribution,
)
from applications.deep_linear.visualize_weights import save_weight_figure
from applications import default_hyperparams
from spiking import (
    Learner,
    STDP,
    WinnerTakesAll,
    CompetitiveThresholdAdaptation,
    TargetTimestampAdaptation,
    SequentialThresholdAdaptation,
    train,
    save_model,
)
from spiking.layers import SpikingSequential

SEED_START = 1
DEFAULT_NUM_SEEDS = 5
T_OBJECTIVES = [round(0.4 + v * 0.05, 2) for v in range(12)] + [0.875]  # 0.4 to 0.95


def train_with_metrics(
    *,
    dataset_loaders: tuple[DataLoader, DataLoader],
    spike_shape: tuple[int, ...],
    seed: int,
    avg_threshold: float,
    output_dir: str,
    num_epochs: int = 30,
    t_objective: float | None = None,
    on_batch_end=None,
    on_epoch_end=None,
) -> dict:
    """Train a single-layer SNN and collect per-neuron training metrics.

    Returns dict with per-neuron lists:
      winner_counts, spike_counts, update_counts,
      threshold_initial, threshold_final, threshold_drift.
    """
    set_seed(seed)
    train_loader, val_loader = dataset_loaders
    num_inputs = math.prod(spike_shape)

    setup = {
        "threshold_init": {
            "avg_threshold": avg_threshold,
            "min_threshold": 1.0,
            "std_dev": 1.0,
        },
    }
    model = create_model(setup, num_inputs, [256])
    layer = model.layers[0]

    adaptation = CompetitiveThresholdAdaptation(
        **default_hyperparams.THRESHOLD_ADAPTATION
    )
    if t_objective is not None:
        adaptation = SequentialThresholdAdaptation(
            [
                adaptation,
                TargetTimestampAdaptation(
                    target_timestamp=t_objective,
                    **default_hyperparams.THRESHOLD_ADAPTATION,
                ),
            ]
        )

    learner = Learner(
        layer,
        learning_mechanism=STDP(**default_hyperparams.STDP),
        competition=WinnerTakesAll(),
        threshold_adaptation=adaptation,
    )

    num_outputs = layer.num_outputs
    win_counts = torch.zeros(num_outputs)
    spike_counts = torch.zeros(num_outputs)
    update_counts = torch.zeros(num_outputs)
    threshold_initial = layer.thresholds.detach().clone().tolist()

    def _tracking_callback(idx, dw, split):
        if split == "train":
            # Track competition winners
            for neuron_idx in learner.neurons_to_learn:
                win_counts[neuron_idx.item()] += 1
                update_counts[neuron_idx.item()] += 1
            # Track all neurons that spiked (finite spike time)
            fired = torch.isfinite(layer.spike_times)
            spike_counts[fired] += 1
        if on_batch_end is not None:
            on_batch_end(idx, dw, split)

    sub_model = SpikingSequential(*model.layers[:1])

    train(
        sub_model,
        learner,
        train_loader,
        num_epochs,
        image_shape=spike_shape,
        on_batch_end=_tracking_callback,
        on_epoch_end=on_epoch_end,
        progress=False,
    )

    threshold_final = layer.thresholds.detach().clone().tolist()
    threshold_drift = [f - i for f, i in zip(threshold_final, threshold_initial)]

    train_m, val_m = evaluate_model(
        sub_model,
        train_loader,
        val_loader,
        t_target=t_objective,
    )

    os.makedirs(output_dir, exist_ok=True)
    save_model(model, f"{output_dir}/model.pth")

    metrics_eval = {"train": train_m, "validation": val_m}
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics_eval, f, indent=4)

    setup_info = {
        "seed": seed,
        "avg_threshold": avg_threshold,
        "num_epochs": num_epochs,
    }
    if t_objective is not None:
        setup_info["t_objective"] = t_objective
    with open(f"{output_dir}/setup.json", "w") as f:
        json.dump(setup_info, f, indent=4)

    training_metrics = {
        "winner_counts": win_counts.tolist(),
        "spike_counts": spike_counts.tolist(),
        "update_counts": update_counts.tolist(),
        "threshold_initial": threshold_initial,
        "threshold_final": threshold_final,
        "threshold_drift": threshold_drift,
    }
    with open(f"{output_dir}/training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=4)

    save_winner_counts(win_counts, f"{output_dir}/winner_counts.png")
    save_threshold_distribution(
        layer.thresholds, f"{output_dir}/threshold_distribution.png"
    )
    save_weight_figure(model.layers[0], spike_shape, f"{output_dir}/weights.png")

    return training_metrics


def run(
    dataset: str,
    *,
    num_epochs: int = 3,
    force: bool = False,
    num_seeds: int = DEFAULT_NUM_SEEDS,
):
    seeds = list(range(SEED_START, SEED_START + num_seeds))
    train_loader, val_loader = create_dataset(dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)
    thresh = math.prod(spike_shape) / 20

    train_steps = len(train_loader)
    val_steps = len(val_loader)
    steps = {"train": train_steps, "val": val_steps}

    total = len(T_OBJECTIVES) * len(seeds)
    with tqdm(total=total, desc="Threshold research: training") as pbar:
        for t_obj in T_OBJECTIVES:
            base_dir = f"logs/{dataset}/threshold_research/tobj_{t_obj}"
            for seed in seeds:
                output_dir = f"{base_dir}/seed_{seed}"
                if not force and os.path.exists(f"{output_dir}/metrics.json"):
                    tqdm.write(f"  skip t_obj={t_obj} seed={seed} (already complete)")
                    pbar.update(1)
                    continue
                label = f"t_obj={t_obj} seed={seed}"
                pbar.set_postfix_str(label)
                on_batch_end, on_epoch_end = make_progress_callbacks(
                    pbar,
                    label,
                    num_epochs,
                    steps,
                )
                train_with_metrics(
                    dataset_loaders=(train_loader, val_loader),
                    spike_shape=spike_shape,
                    seed=seed,
                    avg_threshold=thresh,
                    output_dir=output_dir,
                    num_epochs=num_epochs,
                    t_objective=t_obj,
                    on_batch_end=on_batch_end,
                    on_epoch_end=on_epoch_end,
                )
                pbar.update(1)
            merge_seed_results(base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: Train models with various t_objectives for threshold research"
    )
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--force", action="store_true", help="re-run even if results exist"
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_NUM_SEEDS)
    args = parser.parse_args()

    run(args.dataset, num_epochs=args.epochs, force=args.force, num_seeds=args.seeds)
