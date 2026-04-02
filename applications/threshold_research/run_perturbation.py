import argparse
import json
import os

from tqdm import tqdm

from applications.datasets import DATASETS, create_dataset
from applications.threshold_research.neuron_perturbation import run_perturbation_sweep
from applications.threshold_research.conv_neuron_perturbation import (
    run_conv_perturbation_sweep,
)
from applications.threshold_research.perturbation_params import get_perturbation_params


def _find_models(base_dir: str) -> list[tuple[str, float, int]]:
    """Discover all model paths under base_dir.

    Scans tobj_* and seed_* directories, returning every model found.
    Returns list of (model_path, t_obj, seed) sorted by (t_obj, seed).
    """
    models = []
    if not os.path.isdir(base_dir):
        return models
    for tobj_name in sorted(os.listdir(base_dir)):
        if not tobj_name.startswith("tobj_"):
            continue
        try:
            t_obj = float(tobj_name.split("_", 1)[1])
        except ValueError:
            continue
        tobj_path = os.path.join(base_dir, tobj_name)
        for seed_name in sorted(os.listdir(tobj_path)):
            if not seed_name.startswith("seed_"):
                continue
            try:
                seed = int(seed_name.split("_", 1)[1])
            except ValueError:
                continue
            model_path = os.path.join(tobj_path, seed_name, "model.pth")
            if os.path.exists(model_path):
                models.append((model_path, t_obj, seed))
    return models


def run(
    dataset: str,
    *,
    force: bool = False,
    seeds: list[int] | None = None,
    device: str = "cpu",
    chunk_size: int = 64,
):
    train_loader, val_loader = create_dataset(dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)
    params = get_perturbation_params(dataset)

    base_dir = f"logs/{dataset}/threshold_research"
    models = _find_models(base_dir)
    if seeds:
        models = [(p, t, s) for p, t, s in models if s in seeds]
    if not models:
        print(f"No models found under {base_dir}")
        return

    for model_path, t_obj, seed in tqdm(models, desc="Perturbation sweep"):
        output_path = os.path.join(
            os.path.dirname(model_path), "perturbation_results.json"
        )
        if not force and os.path.exists(output_path):
            tqdm.write(f"  skip t_obj={t_obj} seed={seed} (already complete)")
            continue

        tqdm.write(f"  running t_obj={t_obj} seed={seed}")
        cache_dir = os.path.join(os.path.dirname(model_path), "perturbation_cache")

        if params["is_conv"]:
            result = run_conv_perturbation_sweep(
                model_path=model_path,
                dataset_loaders=(train_loader, val_loader),
                t_target=t_obj,
                pool_size=params["pool_size"],
                seed=seed,
                cache_dir=cache_dir,
                force=force,
                device=device,
                chunk_size=chunk_size,
            )
        else:
            result = run_perturbation_sweep(
                model_path=model_path,
                dataset_loaders=(train_loader, val_loader),
                spike_shape=spike_shape,
                t_target=t_obj,
                seed=seed,
                cache_dir=cache_dir,
                force=force,
            )

        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3: Per-neuron threshold perturbation sweep"
    )
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument(
        "--force", action="store_true", help="re-run even if results exist"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="only run these seeds (default: all)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="device for conv inference"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=64, help="batch chunk size for conv inference"
    )
    args = parser.parse_args()
    run(
        args.dataset,
        force=args.force,
        seeds=args.seeds,
        device=args.device,
        chunk_size=args.chunk_size,
    )
