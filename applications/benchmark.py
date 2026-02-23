import json
import os
from collections.abc import Callable

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from applications.common import set_seed, evaluate_model, aggregate_metrics
from spiking.evaluation import (
    extract_features,
    plot_reduced_features,
    plot_confusion_matrix,
)
from spiking.learning.learner import Learner
from spiking.spiking_module import SpikingModule
from spiking.training import train
from spiking.utils import save_model

DEFAULT_SEEDS = list(range(1, 11))


def _save_seed_plots(model, train_loader, val_loader, image_shape, figures_dir: str):
    from sklearn.svm import LinearSVC

    X_train, y_train = extract_features(model, train_loader, image_shape)
    X_val, y_val = extract_features(model, val_loader, image_shape)

    clf = LinearSVC(max_iter=10000)
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    reducer = plot_reduced_features(X_train, y_train, "Train Data Visualized with PCA")
    plt.subplot(1, 2, 2)
    plot_reduced_features(X_val, y_val, "Val Data Visualized with PCA", reducer=reducer)
    plt.savefig(f"{figures_dir}/reduced_data.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.suptitle("Confusion Matrices")
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(y_train, train_pred)
    plt.subplot(1, 2, 2)
    plot_confusion_matrix(y_val, val_pred)
    plt.savefig(f"{figures_dir}/confusion_matrices.png")
    plt.close()


def benchmark_architecture(
    *,
    create_model: Callable[[], tuple[SpikingModule, Learner]],
    train_loader: DataLoader,
    val_loader: DataLoader,
    image_shape: tuple[int, ...],
    num_epochs: int,
    exp_name: str,
    seeds: list[int] | None = None,
    setup: dict | None = None,
):
    """Train and evaluate an SNN architecture across multiple random seeds.

    Args:
        create_model: Zero-arg factory returning (model, learner). Called after
            set_seed(seed), so random initialization is deterministic per seed.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        image_shape: Full spike volume shape, e.g. (2, 16, 16).
        num_epochs: Number of training epochs per seed.
        exp_name: Experiment name — output goes to logs/{exp_name}/.
        seeds: List of random seeds. Defaults to 1-10.
        setup: Optional metadata dict saved as setup.json per seed.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    all_metrics = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        set_seed(seed)
        model, learner = create_model()

        train(
            model,
            learner,
            train_loader,
            num_epochs,
            image_shape=image_shape,
            val_loader=val_loader,
            progress=True,
        )

        train_metrics, val_metrics = evaluate_model(
            model,
            train_loader,
            val_loader,
            image_shape,
        )

        metrics = {"train": train_metrics, "validation": val_metrics}
        all_metrics.append(metrics)

        # Save per-seed artifacts
        seed_dir = f"logs/{exp_name}/seed_{seed}"
        figures_dir = f"{seed_dir}/figures"
        os.makedirs(figures_dir, exist_ok=True)

        _save_seed_plots(model, train_loader, val_loader, image_shape, figures_dir)

        save_model(model, f"{seed_dir}/model.pth")
        with open(f"{seed_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        if setup is not None:
            with open(f"{seed_dir}/setup.json", "w") as f:
                json.dump(setup, f, indent=4)

        print(f"Seed {seed} — val accuracy: {val_metrics['accuracy']:.4f}")

    # Aggregate across seeds
    exp_dir = f"logs/{exp_name}"
    summary = aggregate_metrics(all_metrics)

    with open(f"{exp_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    with open(f"{exp_dir}/all_seeds_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"\n{'='*60}")
    print(f"Summary ({len(seeds)} seeds)")
    print(f"{'='*60}")
    for split in ("train", "validation"):
        print(f"\n{split}:")
        for metric, stats in summary[split].items():
            print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
