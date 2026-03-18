import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from applications.datasets import create_dataset
from spiking.utils.checkpoints import load_model

RESULTS_DIR_TEMPLATE = "logs/{dataset}/init_cond_research"

_worker_state = {}


def _init_worker(dataset_name):
    """Called once per worker process to load the dataset."""
    train_loader, val_loader = create_dataset(dataset_name)
    _worker_state["train_loader"] = train_loader
    _worker_state["val_loader"] = val_loader


@torch.no_grad()
def compute_spike_stats(
    spike_times: torch.Tensor,
) -> dict:
    """Compute spike statistics from a (N, num_neurons) spike time tensor.

    Neurons that didn't fire have spike_time = inf.
    Returns a dict with the 4 statistics as plain lists.
    """
    n_samples, n_neurons = spike_times.shape
    fired = spike_times < float("inf")

    # 1. First spike time per sample: min across neurons, None if no neuron fired
    min_times, _ = spike_times.min(dim=1)
    any_fired = fired.any(dim=1)
    first_spike_times = [float(t) if f else None for t, f in zip(min_times, any_fired)]

    # 2. Mean and std spike time per neuron (over samples where it fired)
    mean_per_neuron = []
    std_per_neuron = []
    for j in range(n_neurons):
        mask = fired[:, j]
        if mask.any():
            vals = spike_times[mask, j]
            mean_per_neuron.append(float(vals.mean()))
            std_per_neuron.append(float(vals.std()) if vals.numel() > 1 else 0.0)
        else:
            mean_per_neuron.append(None)
            std_per_neuron.append(None)

    # 3. Spike rate per neuron: fraction of samples where each neuron fires
    spike_rate_per_neuron = (fired.sum(dim=0).float() / n_samples).tolist()

    # 4. Spiking neurons per sample: count of neurons that fired
    spiking_neurons_per_sample = fired.sum(dim=1).tolist()

    return {
        "first_spike_times": first_spike_times,
        "mean_spike_time_per_neuron": mean_per_neuron,
        "std_spike_time_per_neuron": std_per_neuron,
        "spike_rate_per_neuron": spike_rate_per_neuron,
        "spiking_neurons_per_sample": spiking_neurons_per_sample,
    }


def _load_full_batch(dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Load entire dataset as a single batch."""
    full_loader = DataLoader(
        dataloader.dataset, batch_size=len(dataloader.dataset), shuffle=False
    )
    return next(iter(full_loader))


def compute_stats_for_model(
    model_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> dict:
    """Load a model and compute spike stats for train and validation splits."""
    model = load_model(model_path)
    model.eval()

    results = {}
    for split_name, loader in [("train", train_loader), ("validation", val_loader)]:
        all_times, _ = _load_full_batch(loader)
        spike_times = model.infer_spike_times_batch(all_times.flatten(1))
        results[split_name] = compute_spike_stats(spike_times)

    return results


def _process_single(model_path, stats_path):
    """Worker function: compute stats for one model and write to disk."""
    stats = compute_stats_for_model(
        model_path,
        _worker_state["train_loader"],
        _worker_state["val_loader"],
    )
    with open(stats_path, "w") as f:
        json.dump(stats, f)


def _collect_tasks(results_dir, force):
    """Collect all (model_path, stats_path) pairs that need processing."""
    tasks = []
    for thresh_dirname in sorted(os.listdir(results_dir)):
        if not thresh_dirname.startswith("thresh_"):
            continue
        thresh_path = os.path.join(results_dir, thresh_dirname)
        for tobj_dirname in sorted(os.listdir(thresh_path)):
            if not tobj_dirname.startswith("tobj_"):
                continue
            tobj_path = os.path.join(thresh_path, tobj_dirname)
            for seed_dirname in sorted(os.listdir(tobj_path)):
                if not seed_dirname.startswith("seed_"):
                    continue
                seed_path = os.path.join(tobj_path, seed_dirname)
                model_path = os.path.join(seed_path, "model.pth")
                stats_path = os.path.join(seed_path, "spike_stats.json")
                if not os.path.exists(model_path):
                    continue
                if not force and os.path.exists(stats_path):
                    continue
                tasks.append((model_path, stats_path))
    return tasks


def run(dataset: str = "mnist", *, force: bool = False, num_workers: int = 1):
    results_dir = RESULTS_DIR_TEMPLATE.format(dataset=dataset)
    if not os.path.isdir(results_dir):
        print(f"No results directory found: {results_dir}")
        return

    tasks = _collect_tasks(results_dir, force)
    if not tasks:
        print("All models already have spike_stats.json (use --force to recompute)")
        return

    print(f"Computing spike stats for {len(tasks)} models ({num_workers} workers)...")

    if num_workers <= 1:
        train_loader, val_loader = create_dataset(dataset)
        for model_path, stats_path in tqdm(tasks, desc="Spike stats"):
            stats = compute_stats_for_model(model_path, train_loader, val_loader)
            with open(stats_path, "w") as f:
                json.dump(stats, f)
    else:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(dataset,),
        ) as pool:
            futures = {
                pool.submit(_process_single, model_path, stats_path): model_path
                for model_path, stats_path in tasks
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Spike stats"
            ):
                future.result()

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute spike time statistics for trained init_cond_research models"
    )
    parser.add_argument(
        "--dataset", default="mnist", help="Dataset name (default: mnist)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute even if spike_stats.json exists"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers"
    )
    args = parser.parse_args()
    run(args.dataset, force=args.force, num_workers=args.workers)
