from applications.common import evaluate_model
from applications.datasets import create_dataset
from applications import default_hyperparams
from applications.deep_linear.model import (
    ARCHITECTURE,
    create_model,
    train_layerwise,
)
from applications.search import find_hyperparameters

# ── User edits these between runs ──────────────────────────────────
NUM_LAYERS = 1  # increase: 1 → 2 → 3
NUM_EPOCHS_PER_LAYER = 50
SEEDS = [1, 2, 3, 4, 5]
DATASET = "mnist"

search_space = {
    "threshold_init": {**default_hyperparams.THRESHOLD_INIT},
    "threshold_adaptation": {
        **default_hyperparams.THRESHOLD_ADAPTATION,
        "learning_rate": [1.0, 5.0, 10.0],
    },
    "stdp": {
        **default_hyperparams.STDP,
        "tau_pre": [0.05, 0.1, 0.2],
        "tau_post": [0.05, 0.1, 0.2],
        "learning_rate": [0.01, 0.05, 0.1],
    },
}
# ───────────────────────────────────────────────────────────────────


def main():
    train_loader, val_loader = create_dataset(DATASET)
    image_shape = train_loader.dataset.image_shape
    spike_shape = (2, *image_shape)
    num_inputs = 2 * image_shape[0] * image_shape[1]

    def run_experiment(config):
        model = create_model(config, num_inputs, ARCHITECTURE)
        train_layerwise(
            model,
            config,
            train_loader,
            val_loader,
            spike_shape,
            num_layers=NUM_LAYERS,
            num_epochs_per_layer=NUM_EPOCHS_PER_LAYER,
        )
        train_metrics, val_metrics = evaluate_model(
            model,
            train_loader,
            val_loader,
        )
        return {"train": train_metrics, "validation": val_metrics}

    find_hyperparameters(
        run_experiment=run_experiment,
        search_space=search_space,
        exp_name=f"linear-deep/hyperparams/layer_{NUM_LAYERS}",
        seeds=SEEDS,
    )


if __name__ == "__main__":
    main()
