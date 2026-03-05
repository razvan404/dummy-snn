import argparse
import json
import os

from tqdm import tqdm

from applications.datasets import DATASETS, create_dataset
from applications.threshold_research.neuron_perturbation import run_perturbation_sweep

T_OBJECTIVES = [round(0.75 + v * 0.05, 2) for v in range(5)]  # 0.75 to 0.95
SEEDS = [1, 2]


def _find_models(base_dir: str) -> list[tuple[str, float, int]]:
    """Find model paths matching selected t_objectives and seeds.

    Returns list of (model_path, t_obj, seed).
    """
    models = []
    for t_obj in T_OBJECTIVES:
        tobj_dir = os.path.join(base_dir, f"tobj_{t_obj}")
        if not os.path.isdir(tobj_dir):
            continue
        for seed in SEEDS:
            model_path = os.path.join(tobj_dir, f"seed_{seed}", "model.pth")
            if os.path.exists(model_path):
                models.append((model_path, t_obj, seed))
    return models


def run(dataset: str, *, force: bool = False):
    train_loader, val_loader = create_dataset(dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    base_dir = f"logs/{dataset}/threshold_research"
    models = _find_models(base_dir)
    if not models:
        print(f"No models found under {base_dir} for t_objectives {T_OBJECTIVES}")
        return

    for model_path, t_obj, seed in tqdm(models, desc="Perturbation sweep"):
        output_path = os.path.join(
            os.path.dirname(model_path), "perturbation_results.json"
        )
        if not force and os.path.exists(output_path):
            tqdm.write(f"  skip t_obj={t_obj} seed={seed} (already complete)")
            continue

        tqdm.write(f"  running t_obj={t_obj} seed={seed}")
        result = run_perturbation_sweep(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=spike_shape,
            t_target=t_obj,
            seed=seed,
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
    args = parser.parse_args()
    run(args.dataset, force=args.force)
