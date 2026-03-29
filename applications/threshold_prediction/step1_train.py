import argparse
import json
import math
import os

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from applications.common import set_seed, evaluate_model
from applications.datasets import DATASETS, create_dataset
from spiking import train, save_model
from spiking.metrics import (
    NeuronTracker,
    compute_vmax_dataset,
    compute_distribution_features,
    compute_inter_neuron_features,
)

from .config import (
    T_OBJECTIVES,
    SEEDS,
    NUM_EPOCHS,
    NUM_OUTPUTS,
    create_model_and_learner,
    output_dir_for,
    save_feature_group,
)


def train_single(
    t_objective: float,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    spike_shape: tuple[int, ...],
    output_dir: str,
    num_epochs: int,
    pbar=None,
) -> dict:
    """Train one model with NeuronTracker and compute all features.

    Saves: model.pth, setup.json, metrics.json, features/*.json.
    Returns evaluation metrics dict.
    """
    set_seed(seed)
    num_inputs = math.prod(spike_shape)
    sub_model, layer, learner = create_model_and_learner(num_inputs, t_objective)

    tracker = NeuronTracker(
        n_neurons=NUM_OUTPUTS,
        n_epochs=num_epochs,
        n_classes=10,
        weight_snapshot_interval=10,
    )

    current_epoch = [0]
    label_cache = [None]

    def on_batch_end(idx, dw, split):
        if split == "train":
            tracker.on_batch(learner, label=label_cache[0], dw=dw)
        if pbar and idx % 500 == 0:
            pbar.set_postfix_str(
                f"tobj={t_objective} seed={seed} "
                f"epoch={current_epoch[0] + 1}/{num_epochs} {split} {idx + 1}"
            )

    def on_epoch_end(epoch, total):
        tracker.on_epoch_end(current_epoch[0], layer)
        current_epoch[0] = epoch
        if epoch < num_epochs:
            tracker.on_epoch_start(epoch, layer)

    # Wrap dataset to capture per-sample labels for the tracker
    class LabelCachingDataset:
        def __init__(self, dataset, label_ref):
            self.dataset = dataset
            self.label_ref = label_ref

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            times, label = self.dataset[idx]
            self.label_ref[0] = label if isinstance(label, int) else label.item()
            return times, label

    cached_train = LabelCachingDataset(train_loader.dataset, label_cache)
    cached_loader = DataLoader(cached_train, batch_size=None, shuffle=True)

    tracker.on_epoch_start(0, layer)

    train(
        sub_model,
        learner,
        cached_loader,
        num_epochs,
        image_shape=spike_shape,
        val_loader=val_loader,
        on_batch_end=on_batch_end,
        on_epoch_end=on_epoch_end,
        progress=False,
    )

    # --- Evaluate ---
    train_m, val_m = evaluate_model(
        sub_model, train_loader, val_loader, t_target=t_objective
    )
    tqdm.write(
        f"  tobj={t_objective} seed={seed}: "
        f"train={train_m['accuracy']:.4f} val={val_m['accuracy']:.4f}"
    )

    # --- Compute features ---
    layer = layer.cpu()
    layer.eval()
    tqdm.write("  Computing V_max + features...")
    V_max_train, spike_train, _ = compute_vmax_dataset(layer, train_loader)

    thresholds = layer.thresholds.detach().cpu().numpy()
    weights = layer.weights.detach().cpu().numpy()

    trajectory = tracker.compute_trajectory_features()
    distribution = compute_distribution_features(V_max_train, thresholds)
    inter_neuron = compute_inter_neuron_features(
        weights, V_max_train, spike_train, thresholds
    )

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    save_model(sub_model, f"{output_dir}/model.pth")

    with open(f"{output_dir}/setup.json", "w") as f:
        json.dump(
            {
                "seed": seed,
                "t_objective": t_objective,
                "num_epochs": num_epochs,
                "num_outputs": NUM_OUTPUTS,
            },
            f,
            indent=4,
        )

    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump({"train": train_m, "validation": val_m}, f, indent=4)

    feat_dir = f"{output_dir}/features"
    os.makedirs(feat_dir, exist_ok=True)
    save_feature_group(f"{feat_dir}/trajectory.json", trajectory)
    save_feature_group(f"{feat_dir}/distribution.json", distribution)
    save_feature_group(f"{feat_dir}/inter_neuron.json", inter_neuron)
    np.save(f"{feat_dir}/trained_thresholds.npy", thresholds)

    return {"train": train_m, "validation": val_m}


def run(
    dataset: str,
    *,
    num_epochs: int = NUM_EPOCHS,
    force: bool = False,
    seeds: list[int] | None = None,
):
    train_loader, val_loader = create_dataset(dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)
    if seeds is None:
        seeds = SEEDS

    base_dir = f"logs/{dataset}/threshold_prediction"
    total = len(T_OBJECTIVES) * len(seeds)

    pbar = tqdm(total=total, desc="Step 1: Train + features")
    for t_obj in T_OBJECTIVES:
        for seed in seeds:
            out = output_dir_for(base_dir, t_obj, seed)
            pbar.set_postfix_str(f"tobj={t_obj} seed={seed}")

            if not force and os.path.exists(f"{out}/features/trained_thresholds.npy"):
                tqdm.write(f"  skip tobj={t_obj} seed={seed} (already complete)")
                pbar.update(1)
                continue

            train_single(
                t_obj,
                seed,
                train_loader,
                val_loader,
                spike_shape,
                out,
                num_epochs,
                pbar,
            )
            pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: Train models and collect training/post-hoc metrics"
    )
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument(
        "--force", action="store_true", help="Re-run even if results exist"
    )
    parser.add_argument("--seeds", type=int, nargs="+", help="Only run these seeds")
    args = parser.parse_args()
    run(args.dataset, num_epochs=args.epochs, force=args.force, seeds=args.seeds)
