import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm

from applications.common import set_seed, evaluate_model, merge_seed_results
from applications.datasets import DATASETS, create_dataset
from applications import default_hyperparams
from applications.deep_linear.model import create_model, ARCHITECTURE
from applications.deep_linear.training_plots import (
    save_winner_counts,
    save_threshold_distribution,
)
from applications.deep_linear.visualize_weights import save_weight_figure
from spiking import (
    Learner,
    STDP,
    WinnerTakesAll,
    CompetitiveThresholdAdaptation,
    train,
    save_model,
)
from spiking.layers import SpikingSequential

SEED_START = 1
DEFAULT_NUM_SEEDS = 5


def _save_plots(dynamics, activity, win_counts, layer, spike_shape, figures_dir):
    """Generate diagnostic plots for training dynamics."""
    os.makedirs(figures_dir, exist_ok=True)

    # Weight evolution: scatter + moving average
    dw = dynamics["weight_diffs"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(dw)), dw, s=1, alpha=0.3, label="per-batch")
    if len(dw) >= 20:
        window = max(1, len(dw) // 20)
        kernel = torch.ones(window) / window
        smoothed = torch.nn.functional.conv1d(
            torch.tensor(dw).unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=window // 2,
        ).squeeze()
        ax.plot(range(len(smoothed)), smoothed.numpy(), color="red", label="moving avg")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Weight diff")
    ax.set_title("Weight evolution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{figures_dir}/weight_evolution.png", dpi=100)
    plt.close(fig)

    # Threshold evolution: mean/min/max
    fig, ax = plt.subplots(figsize=(10, 4))
    batches = range(len(dynamics["thresholds_mean"]))
    ax.plot(batches, dynamics["thresholds_mean"], label="mean")
    ax.plot(batches, dynamics["thresholds_min"], label="min")
    ax.plot(batches, dynamics["thresholds_max"], label="max")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Threshold")
    ax.set_title("Threshold evolution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{figures_dir}/threshold_evolution.png", dpi=100)
    plt.close(fig)

    # Neuron activity: bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(activity)), activity.numpy())
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Spike count")
    ax.set_title("Neuron activity")
    fig.tight_layout()
    fig.savefig(f"{figures_dir}/neuron_activity.png", dpi=100)
    plt.close(fig)

    save_winner_counts(win_counts, f"{figures_dir}/winner_counts.png")
    save_threshold_distribution(
        layer.thresholds, f"{figures_dir}/threshold_distribution.png"
    )

    save_weight_figure(layer, spike_shape, f"{figures_dir}/weights.png")


def run(
    dataset: str,
    *,
    num_epochs: int = 10,
    force: bool = False,
    num_seeds: int = DEFAULT_NUM_SEEDS,
):
    seeds = list(range(SEED_START, SEED_START + num_seeds))
    train_loader, val_loader = create_dataset(dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)
    num_inputs = math.prod(spike_shape)
    avg_threshold = num_inputs / 20

    base_dir = f"logs/{dataset}/layer_1/params_establish"
    train_steps = len(train_loader)

    with tqdm(total=len(seeds), desc="Exp 0: params establish") as pbar:
        for seed in seeds:
            output_dir = f"{base_dir}/seed_{seed}"
            if not force and os.path.exists(f"{output_dir}/metrics.json"):
                tqdm.write(f"  skip seed={seed} (already complete)")
                pbar.update(1)
                continue
            pbar.set_postfix_str(f"seed={seed}")
            set_seed(seed)

            setup = {
                "threshold_init": {
                    "avg_threshold": avg_threshold,
                    "min_threshold": 1.0,
                    "std_dev": 1.0,
                },
            }
            model = create_model(setup, num_inputs)
            layer = model.layers[0]
            sub_model = SpikingSequential(layer)

            learner = Learner(
                layer,
                learning_mechanism=STDP(**default_hyperparams.STDP),
                competition=WinnerTakesAll(),
                threshold_adaptation=CompetitiveThresholdAdaptation(
                    **default_hyperparams.THRESHOLD_ADAPTATION
                ),
            )

            dynamics = {
                "weight_diffs": [],
                "thresholds_mean": [],
                "thresholds_min": [],
                "thresholds_max": [],
            }
            activity = torch.zeros(layer.num_outputs)
            win_counts = torch.zeros(layer.num_outputs)
            epoch = [1]

            def on_batch_end(idx, dw, split):
                if split != "train":
                    return
                dynamics["weight_diffs"].append(dw)
                if idx % 200 == 0:
                    dynamics["thresholds_mean"].append(layer.thresholds.mean().item())
                    dynamics["thresholds_min"].append(layer.thresholds.min().item())
                    dynamics["thresholds_max"].append(layer.thresholds.max().item())
                activity[torch.isfinite(layer.spike_times)] += 1
                for neuron_idx in learner.neurons_to_learn:
                    win_counts[neuron_idx.item()] += 1
                if idx % 200 == 0:
                    pbar.set_postfix_str(
                        f"seed={seed} epoch={epoch[0]}/{num_epochs} {idx+1}/{train_steps} dw={dw:.4f}"
                    )

            train(
                sub_model,
                learner,
                train_loader,
                num_epochs,
                image_shape=spike_shape,
                on_batch_end=on_batch_end,
                on_epoch_end=lambda e, _total: epoch.__setitem__(0, e + 1),
                progress=False,
            )

            train_m, val_m = evaluate_model(sub_model, train_loader, val_loader)

            os.makedirs(output_dir, exist_ok=True)
            save_model(model, f"{output_dir}/model.pth")

            metrics = {"train": train_m, "validation": val_m}
            with open(f"{output_dir}/metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            setup_info = {
                "seed": seed,
                "avg_threshold": avg_threshold,
                "num_inputs": num_inputs,
                "num_epochs": num_epochs,
            }
            with open(f"{output_dir}/setup.json", "w") as f:
                json.dump(setup_info, f, indent=4)

            dynamics["win_counts"] = win_counts.tolist()
            with open(f"{output_dir}/dynamics.json", "w") as f:
                json.dump(dynamics, f, indent=4)

            _save_plots(
                dynamics,
                activity,
                win_counts,
                layer,
                spike_shape,
                f"{output_dir}/figures",
            )

            pbar.update(1)

    merge_seed_results(base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 0: establish baseline parameters with training dynamics"
    )
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--force", action="store_true", help="re-run even if results exist"
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_NUM_SEEDS)
    args = parser.parse_args()

    run(args.dataset, num_epochs=args.epochs, force=args.force, num_seeds=args.seeds)
