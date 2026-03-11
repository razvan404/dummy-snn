# ABOUTME: Pre-computes post-hoc per-neuron metrics for all trained models.
# ABOUTME: Iterates tobj_*/seed_* directories and saves post_hoc_metrics.json per seed.
import argparse
import json
import os

from tqdm import tqdm

from applications.datasets import DATASETS, create_dataset
from applications.threshold_research.analysis import compute_post_hoc_metrics
from applications.threshold_research.run_perturbation import _find_models


def run(dataset: str, *, force: bool = False, seeds: list[int] | None = None):
    train_loader, val_loader = create_dataset(dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    base_dir = f"logs/{dataset}/threshold_research"
    models = _find_models(base_dir)
    if seeds:
        models = [(p, t, s) for p, t, s in models if s in seeds]
    if not models:
        print(f"No models found under {base_dir}")
        return

    for model_path, t_obj, seed in tqdm(models, desc="Post-hoc metrics"):
        output_path = os.path.join(
            os.path.dirname(model_path), "post_hoc_metrics.json"
        )
        if not force and os.path.exists(output_path):
            tqdm.write(f"  skip t_obj={t_obj} seed={seed} (already complete)")
            continue

        tqdm.write(f"  running t_obj={t_obj} seed={seed}")
        metrics = compute_post_hoc_metrics(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=spike_shape,
        )

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute post-hoc per-neuron metrics for all trained models"
    )
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument(
        "--force", action="store_true", help="re-run even if results exist"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="only run these seeds (default: all)"
    )
    args = parser.parse_args()
    run(args.dataset, force=args.force, seeds=args.seeds)
