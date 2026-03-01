import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader

from applications.datasets import create_dataset
from applications.single_layer.evaluate import eval_snn
from applications.single_layer.model import create_model_and_learner, get_default_setup
from applications.single_layer.visualize import save_visualizations
from spiking.training import TrainingMonitor, train
from spiking.utils import save_model


def load_datasets():
    train_loader, test_loader = create_dataset("mnist_subset")
    image_shape = train_loader.dataset.image_shape
    num_inputs = 2 * image_shape[0] * image_shape[1]
    spike_shape = (2, *image_shape)

    train_dataset, val_dataset = random_split(train_loader.dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False)

    return train_loader, val_loader, test_loader, num_inputs, spike_shape


def train_one_layer(exp_name: str, setup: dict):
    np.random.seed(setup["seed"])
    torch.manual_seed(setup["seed"])

    train_loader, val_loader, test_loader, num_inputs, spike_shape = load_datasets()
    num_outputs = setup["num_outputs"]

    model, learner = create_model_and_learner(setup, num_inputs, num_outputs)

    logs_dir = f"logs/{exp_name}"
    figures_dir = f"{logs_dir}/figures"

    os.makedirs(figures_dir, exist_ok=True)
    monitor = TrainingMonitor(model, log_interval=100)

    train(
        model,
        learner,
        train_loader,
        setup["num_epochs"],
        image_shape=spike_shape,
        on_batch_end=lambda batch_idx, dw, split: monitor.log(split=split, dw=dw),
    )

    model = model.cpu()
    save_visualizations(monitor, figures_dir, spike_shape[1:])
    train_metrics, val_metrics = eval_snn(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        spike_shape=spike_shape,
        figures_dir=figures_dir,
    )

    with open(f"{logs_dir}/setup.json", "w") as f:
        json.dump(setup, f, indent=4)
    with open(f"{logs_dir}/metrics.json", "w") as f:
        json.dump({"train": train_metrics, "validation": val_metrics}, f, indent=4)
    with open(f"{logs_dir}/thresholds.json", "w") as f:
        json.dump(monitor.thresholds["list"], f)

    save_model(model, f"{logs_dir}/model.pth")
    print(f"Experiment {exp_name} finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single-layer SNN")
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="Experiment name (used for output directory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--num-outputs",
        type=int,
        default=100,
        help="Number of output neurons (default: 100)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    args = parser.parse_args()

    setup = get_default_setup(
        seed=args.seed,
        num_outputs=args.num_outputs,
        num_epochs=args.num_epochs,
    )
    train_one_layer(exp_name=args.exp_name, setup=setup)
