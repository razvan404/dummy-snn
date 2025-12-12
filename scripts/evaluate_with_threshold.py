import json
import os

import numpy as np
import torch
import tqdm

from spiking.layers import IntegrateAndFireLayer
from spiking.utils import load_model
from scripts.train_one_layer import eval_snn, load_datasets


@torch.no_grad()
def evaluate_with_threshold(setup: dict):
    np.random.seed(setup["seed"])
    torch.manual_seed(setup["seed"])

    model_path = f"logs/{setup['model_name']}/model.pth"
    logs_path = f"logs/{setup['exp_name']}"
    os.makedirs(logs_path, exist_ok=True)

    train_loader, val_loader, _, num_inputs = load_datasets()
    model: IntegrateAndFireLayer = load_model(model_path)

    with open(f"{logs_path}/setup.json", "w") as f:
        json.dump(setup, f)

    for threshold in tqdm.tqdm(
        np.linspace(setup["threshold_min"], setup["threshold_max"], setup["num_steps"])
    ):
        model.thresholds.fill_(threshold)
        train_metrics, val_metrics = eval_snn(
            model, train_loader=train_loader, val_loader=val_loader, visualize=False
        )

        with open(f"{logs_path}/metrics_{threshold:.2f}.json", "w") as f:
            json.dump({"train": train_metrics, "validation": val_metrics}, f)


def main():
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


if __name__ == "__main__":
    main()
