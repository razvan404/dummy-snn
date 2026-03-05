import os
import sys
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

from applications.common import merge_seed_results


def find_trained_models(
    base_dir: Path, *, exclude_dirs: tuple[str, ...] = ()
) -> list[Path]:
    """Find model.pth files under comp_only/ and comp_t_obj/, excluding post-training subdirs."""
    always_exclude = ("pbtr", "random_thresh", "uniform_thresh")
    all_exclude = always_exclude + exclude_dirs
    return [
        p
        for group in ("comp_only", "comp_t_obj")
        for p in sorted(base_dir.glob(f"{group}/*/*/model.pth"))
        if not any(part.startswith(ex) for part in p.parts for ex in all_exclude)
    ]


def sweep_post_training(
    *,
    model_paths: list[Path],
    seeds: list[int],
    fn: Callable[[str, str, int], None],
    result_subdir: str,
    description: str,
    force: bool = False,
):
    """Run fn(model_path, output_dir, seed) for each model × seed combination.

    fn should produce metrics.json in output_dir.
    result_subdir is the name of the directory under each model's seed dir (e.g. "random_thresh").
    """
    if not model_paths:
        print(f"No models found.")
        sys.exit(0)

    total = len(model_paths) * len(seeds)
    with tqdm(total=total, desc=description) as pbar:
        for model_path in model_paths:
            seed_dir = model_path.parent
            ctrl_dir = seed_dir / result_subdir
            for seed in seeds:
                output_dir = str(ctrl_dir / f"seed_{seed}")
                if not force and os.path.exists(f"{output_dir}/metrics.json"):
                    tqdm.write(f"  skip {seed_dir.name} seed={seed} (already complete)")
                    pbar.update(1)
                    continue
                pbar.set_postfix_str(f"{seed_dir.name} seed={seed}")
                fn(str(model_path), output_dir, seed)
                pbar.update(1)
            merge_seed_results(str(ctrl_dir))
