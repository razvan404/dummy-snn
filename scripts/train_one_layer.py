import json

import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import random_split, DataLoader

from datasets.mnist import MnistDataset
from spiking.evaluation import SpikingClassifierEvaluator
from spiking.layers import IntegrateAndFireLayer
from spiking.registry import registry
from spiking.training.if_callbacks import IntegrateAndFireCallbacks
from spiking.training.monitor import TrainingMonitor
from spiking.training.unsupervised_trainer import UnsupervisedTrainer
from spiking.utils import save_model

IMAGE_SHAPE = (16, 16)


def create_model(setup: dict, num_inputs: int, num_outputs: int):
    def load(name: str):
        return registry.create(name, *setup[name])

    threshold_initialization = load("threshold.initialization")
    threshold_adaptation = load("threshold.adaptation")
    learning_mechanism = load("learning_mechanism")
    competition_mechanism = load("competition_mechanism")

    return IntegrateAndFireLayer(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        learning_mechanism=learning_mechanism,
        threshold_initialization=threshold_initialization,
        competition_mechanism=competition_mechanism,
        threshold_adaptation=threshold_adaptation,
        refractory_period=np.inf,
    )


def load_datasets():
    dataset = MnistDataset("data/mnist-subset", "train", image_shape=IMAGE_SHAPE)
    test_dataloader = MnistDataset("data/mnist-subset", "test", image_shape=IMAGE_SHAPE)

    max_x, max_y = IMAGE_SHAPE
    max_z = 2
    max_input_spikes = max_x * max_y * max_z

    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False)
    test_loader = DataLoader(test_dataloader, batch_size=None, shuffle=False)

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
    figures_dir: str | None = None,
    verbose: bool = False,
):
    evaluator = SpikingClassifierEvaluator(
        model, train_loader, val_loader, shape=(2, *IMAGE_SHAPE)
    )
    print(f"{evaluator.X_train.shape = }, {evaluator.y_train.shape = }")
    print(f"{evaluator.X_test.shape = }, {evaluator.y_test.shape = }")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    reducer = evaluator.plot_reduced_dataset("train")
    plt.subplot(1, 2, 2)
    evaluator.plot_reduced_dataset("val", reducer=reducer)

    if figures_dir is not None:
        plt.savefig(f"{figures_dir}/reduced_data.png")
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(12, 5))
    plt.suptitle("Confusion Matrices")
    train_metrics, val_metrics = evaluator.eval_classifier(
        classifier=classifier, train=train, visualize=False, verbose=verbose
    )

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

    model = create_model(setup, num_inputs, num_outputs)

    logs_dir = f"logs/{exp_name}"
    figures_dir = f"{logs_dir}/figures"

    callbacks = IntegrateAndFireCallbacks(model, figures_dir=figures_dir)

    for epoch in tqdm.trange(setup["num_epochs"]):
        callbacks.epoch = epoch

        trainer = UnsupervisedTrainer(
            model, image_shape=(2, *IMAGE_SHAPE), callbacks=callbacks
        )
        trainer.step_loader(train_loader, split="train")
        trainer.step_loader(val_loader, split="val")
        trainer.step_epoch()

    model = trainer.model.cpu()
    save_visualizations(callbacks.monitor, figures_dir)
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

    save_model(model, f"{logs_dir}/model.pth")
    print(f"Experiment {exp_name} finished.")


if __name__ == "__main__":
    decay_lambda = 1.0
    weights_lr0 = 0.1
    adapt_lr0 = 5.0

    tau = 0.1
    max_pre_spike_time = 1.0

    min_threshold = 1.0
    avg_threshold = 25.0
    std_dev_threshold = 1.0

    train_one_layer(
        exp_name="test",
        setup={
            "dataset": "mnist",
            "num_epochs": 20,
            "num_outputs": 36,
            "seed": 42,
            "threshold.initialization": (
                "normal",
                {
                    "avg_threshold": avg_threshold,
                    "min_threshold": min_threshold,
                    "std_dev": std_dev_threshold,
                },
            ),
            "threshold.adaptation": (
                "competitive_falez",
                {
                    "min_threshold": min_threshold,
                    "learning_rate": adapt_lr0,
                    "decay_factor": decay_lambda,
                },
            ),
            "learning_mechanism": (
                "stdp",
                {
                    "tau_pre": tau,
                    "tau_post": tau,
                    "max_pre_spike_time": max_pre_spike_time,
                    "learning_rate": weights_lr0,
                    "decay_factor": decay_lambda,
                },
            ),
            "competition_mechanism": ("wta", {}),
        },
    )
