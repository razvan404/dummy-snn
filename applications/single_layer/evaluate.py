import json
import os

import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from spiking.evaluation import (
    extract_features,
    compute_metrics,
    plot_confusion_matrix,
    plot_reduced_features,
)
from spiking.layers import IntegrateAndFireLayer
from spiking.utils import load_model


def eval_snn(
    model: IntegrateAndFireLayer,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    spike_shape: tuple[int, ...],
    classifier=None,
    train: bool = True,
    visualize: bool = True,
    figures_dir: str | None = None,
    verbose: bool = False,
):
    from sklearn.svm import LinearSVC

    X_train, y_train = extract_features(model, train_loader)
    X_test, y_test = extract_features(model, val_loader)

    if verbose:
        print(f"{X_train.shape = }, {y_train.shape = }")
        print(f"{X_test.shape = }, {y_test.shape = }")

    if visualize:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        reducer = plot_reduced_features(
            X_train, y_train, "Train Data Visualized with PCA"
        )
        plt.subplot(1, 2, 2)
        plot_reduced_features(
            X_test, y_test, "Val Data Visualized with PCA", reducer=reducer
        )

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


@torch.no_grad()
def evaluate_with_threshold(setup: dict):
    np.random.seed(setup["seed"])
    torch.manual_seed(setup["seed"])

    model_path = f"logs/{setup['model_name']}/model.pth"
    logs_path = f"logs/{setup['exp_name']}"
    os.makedirs(logs_path, exist_ok=True)

    from applications.single_layer.train import load_datasets

    train_loader, val_loader, _, num_inputs, spike_shape = load_datasets()
    model: IntegrateAndFireLayer = load_model(model_path)

    with open(f"{logs_path}/setup.json", "w") as f:
        json.dump(setup, f)

    for threshold in tqdm.tqdm(
        np.linspace(setup["threshold_min"], setup["threshold_max"], setup["num_steps"])
    ):
        model.thresholds.fill_(threshold)
        train_metrics, val_metrics = eval_snn(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            spike_shape=spike_shape,
            visualize=False,
        )

        with open(f"{logs_path}/metrics_{threshold:.2f}.json", "w") as f:
            json.dump({"train": train_metrics, "validation": val_metrics}, f)


if __name__ == "__main__":
    threshold = 35
    seed = 42
    model_name = f"model_th{threshold}_seed{seed}"
    num_steps = 10

    setup = {
        "dataset": "mnist",
        "seed": seed,
        "exp_name": f"{model_name}_linspace",
        "model_name": model_name,
        "threshold_min": 5,
        "threshold_max": 10,
        "num_steps": num_steps,
    }
    evaluate_with_threshold(setup)
