import argparse
import json
import os

import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import random_split, DataLoader

from scripts.common import create_dataset
from spiking.evaluation import (
    extract_features,
    compute_metrics,
    plot_confusion_matrix,
    plot_reduced_features,
)
from spiking.layers import IntegrateAndFireLayer
from spiking.learning import Learner
from spiking.registry import registry
from spiking.training import UnsupervisedTrainer, TrainingMonitor
from spiking.utils import save_model

IMAGE_SHAPE = (16, 16)


def create_model_and_learner(setup: dict, num_inputs: int, num_outputs: int):
    def load(name: str):
        return registry.create(name, *setup[name])

    threshold_initialization = load("threshold.initialization")
    threshold_adaptation = load("threshold.adaptation")
    learning_mechanism = load("learning_mechanism")
    competition_mechanism = load("competition_mechanism")

    model = IntegrateAndFireLayer(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        threshold_initialization=threshold_initialization,
        refractory_period=np.inf,
    )
    learner = Learner(
        model,
        learning_mechanism,
        competition=competition_mechanism,
        threshold_adaptation=threshold_adaptation,
    )
    return model, learner


def load_datasets():
    train_loader, test_loader = create_dataset("mnist_subset", IMAGE_SHAPE)

    max_x, max_y = IMAGE_SHAPE
    max_z = 2
    max_input_spikes = max_x * max_y * max_z

    train_dataset, val_dataset = random_split(train_loader.dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False)

    return train_loader, val_loader, test_loader, max_input_spikes


def save_visualizations(monitor: TrainingMonitor, figures_dir: str):
    plt.figure(figsize=(12, 5))
    plt.suptitle("Weights evolution")
    plt.subplot(1, 2, 1)
    monitor.plot_weight_evolution("train", title="Train split")
    plt.subplot(1, 2, 2)
    monitor.plot_weight_evolution("val", title="Validation split")
    plt.savefig(f"{figures_dir}/weight_evolution.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.suptitle("Network evolution")
    plt.subplot(1, 2, 1)
    monitor.plot_thresholds_evolution(title="Thresholds evolution")
    plt.subplot(1, 2, 2)
    monitor.plot_neurons_activity()
    plt.savefig(f"{figures_dir}/network_evolution.png")
    plt.close()

    monitor.visualize_weights(
        IMAGE_SHAPE,
        monitor.most_active_neurons(min(32, monitor.model.num_outputs)),
        ncols=8,
    )
    plt.savefig(f"{figures_dir}/weights.png")
    plt.close()


def eval_snn(
    model: IntegrateAndFireLayer,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    classifier=None,
    train: bool = True,
    visualize: bool = True,
    figures_dir: str | None = None,
    verbose: bool = False,
):
    from sklearn.svm import LinearSVC

    shape = (2, *IMAGE_SHAPE)
    X_train, y_train = extract_features(model, train_loader, shape)
    X_test, y_test = extract_features(model, val_loader, shape)

    if verbose:
        print(f"{X_train.shape = }, {y_train.shape = }")
        print(f"{X_test.shape = }, {y_test.shape = }")

    if visualize:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        reducer = plot_reduced_features(X_train, y_train, "Train Data Visualized with PCA")
        plt.subplot(1, 2, 2)
        plot_reduced_features(X_test, y_test, "Val Data Visualized with PCA", reducer=reducer)

        if figures_dir is not None:
            plt.savefig(f"{figures_dir}/reduced_data.png")
            plt.close()
        else:
            plt.show()

    if classifier is None:
        classifier = LinearSVC(max_iter=20000)
    if train:
        classifier.fit(X_train, y_train)

    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)
    train_metrics = compute_metrics(y_train, y_train_pred)
    val_metrics = compute_metrics(y_test, y_test_pred)

    if verbose:
        print("Train metrics:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.2f}")
        print("Validation metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.2f}")

    if visualize:
        plt.figure(figsize=(12, 5))
        plt.suptitle("Confusion Matrices")
        plt.subplot(1, 2, 1)
        plot_confusion_matrix(y_train, y_train_pred)
        plt.subplot(1, 2, 2)
        plot_confusion_matrix(y_test, y_test_pred)

        if figures_dir is not None:
            plt.savefig(f"{figures_dir}/confusion_matrices.png")
            plt.close()
        else:
            plt.show()

    return train_metrics, val_metrics


def train_one_layer(exp_name: str, setup: dict):
    np.random.seed(setup["seed"])
    torch.manual_seed(setup["seed"])

    train_loader, val_loader, test_loader, num_inputs = load_datasets()
    num_outputs = setup["num_outputs"]

    model, learner = create_model_and_learner(setup, num_inputs, num_outputs)

    logs_dir = f"logs/{exp_name}"
    figures_dir = f"{logs_dir}/figures"

    os.makedirs(figures_dir, exist_ok=True)
    monitor = TrainingMonitor(model)

    for epoch in tqdm.trange(setup["num_epochs"]):
        trainer = UnsupervisedTrainer(
            model, learner, image_shape=(2, *IMAGE_SHAPE), monitor=monitor
        )
        trainer.step_loader(train_loader, split="train")
        trainer.step_loader(val_loader, split="val")
        trainer.step_epoch()

    model = trainer.model.cpu()
    save_visualizations(monitor, figures_dir)
    train_metrics, val_metrics = eval_snn(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
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


def get_default_setup(seed: int, num_outputs: int = 100, num_epochs: int = 100):
    """Return default training setup with given seed."""
    return {
        "dataset": "mnist_subset",
        "num_epochs": num_epochs,
        "num_outputs": num_outputs,
        "seed": seed,
        "threshold.initialization": (
            "normal",
            {
                "avg_threshold": 25.6,
                "min_threshold": 1.0,
                "std_dev": 1.0,
            },
        ),
        "threshold.adaptation": (
            "competitive",
            {
                "min_threshold": 1.0,
                "learning_rate": 5.0,
                "decay_factor": 1.0,
            },
        ),
        "learning_mechanism": (
            "stdp",
            {
                "tau_pre": 0.1,
                "tau_post": 0.1,
                "max_pre_spike_time": 1.0,
                "learning_rate": 0.1,
                "decay_factor": 1.0,
            },
        ),
        "competition_mechanism": ("wta", {}),
    }


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
