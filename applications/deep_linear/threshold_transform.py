import json
import os
from collections.abc import Callable

from torch.utils.data import DataLoader

from applications.common import set_seed, evaluate_model
from spiking import load_model, save_model
from spiking.layers import SpikingSequential
from spiking.layers.integrate_and_fire import IntegrateAndFireLayer


def apply_threshold_transform(
    transform_fn: Callable[[IntegrateAndFireLayer], None],
    *,
    model_path: str,
    dataset_loaders: tuple[DataLoader, DataLoader],
    spike_shape: tuple[int, ...],
    seed: int,
    output_dir: str,
    layer_idx: int = 0,
):
    """Load a model, apply transform_fn to a layer's thresholds, evaluate, and save.

    transform_fn receives the layer and should modify layer.thresholds.data in place.
    """
    set_seed(seed)
    train_loader, val_loader = dataset_loaders

    model = load_model(model_path)
    layer = model.layers[layer_idx]
    transform_fn(layer)

    sub_model = SpikingSequential(*model.layers[: layer_idx + 1])
    train_m, val_m = evaluate_model(sub_model, train_loader, val_loader, spike_shape)

    os.makedirs(output_dir, exist_ok=True)
    save_model(model, f"{output_dir}/model.pth")
    metrics = {"train": train_m, "validation": val_m}
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
