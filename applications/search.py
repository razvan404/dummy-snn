import copy
import itertools
import json
import os
from collections.abc import Callable

import tqdm

from applications.common import set_seed, aggregate_metrics

DEFAULT_SEEDS = [1, 2, 3, 4, 5]


def resolve_search_space(
    search_space: dict,
) -> tuple[list[tuple[str, str]], list[dict]]:
    """Enumerate all grid combinations of list-valued kwargs in the setup dict.

    Returns (tunable_param_names, resolved_configs).

    tunable_param_names: list of (component_key, param_name) for display.
    resolved_configs: list of setup dicts with all lists resolved to scalars.
    """
    tunable_params = []  # (component_key, param_name, [values])

    for component_key, kwargs in search_space.items():
        if not isinstance(kwargs, dict):
            continue
        for param_name, value in kwargs.items():
            if isinstance(value, list):
                tunable_params.append((component_key, param_name, value))

    param_names = [(comp, param) for comp, param, _ in tunable_params]

    if not tunable_params:
        return param_names, [search_space]

    value_lists = [values for _, _, values in tunable_params]

    configs = []
    for combo in itertools.product(*value_lists):
        config = copy.deepcopy(search_space)
        for (comp_key, param_name, _), value in zip(tunable_params, combo):
            config[comp_key][param_name] = value
        configs.append(config)

    return param_names, configs


def find_hyperparameters(
    *,
    run_experiment: Callable[[dict], dict],
    search_space: dict,
    exp_name: str,
    seeds: list[int] | None = None,
) -> dict:
    """Grid search over hyperparameters, returning the best resolved config.

    Args:
        run_experiment: Callable taking a resolved config dict and returning
            {"train": {...}, "validation": {...}} metrics dict.
        search_space: Setup dict where list-valued kwargs define the search grid.
        exp_name: Experiment name — output goes to logs/{exp_name}/.
        seeds: Random seeds per config. Defaults to [1, 2, 3, 4, 5].
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    param_names, configs = resolve_search_space(search_space)
    print(f"Search space: {len(configs)} configs × {len(seeds)} seeds")
    if param_names:
        print(f"Tunable params: {', '.join(f'{c}.{p}' for c, p in param_names)}")

    results = []

    for config_idx, config in enumerate(configs):
        tunable_values = {
            f"{comp}.{param}": config[comp][param] for comp, param in param_names
        }
        print(f"\n{'='*60}")
        print(f"Config {config_idx + 1}/{len(configs)}: {tunable_values}")
        print(f"{'='*60}")

        all_metrics = []
        for seed in tqdm.tqdm(seeds, desc=f"Config {config_idx + 1}"):
            set_seed(seed)
            metrics = run_experiment(config)
            all_metrics.append(metrics)

        summary = aggregate_metrics(all_metrics)
        results.append({"config": config, "metrics": summary})

        mean_acc = summary["validation"]["accuracy"]["mean"]
        std_acc = summary["validation"]["accuracy"]["std"]
        print(f"  val accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    # Sort by mean validation accuracy (descending)
    results.sort(
        key=lambda r: r["metrics"]["validation"]["accuracy"]["mean"],
        reverse=True,
    )

    # Save results
    exp_dir = f"logs/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    with open(f"{exp_dir}/search_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Results (sorted by validation accuracy)")
    print(f"{'='*60}")
    for i, result in enumerate(results):
        val_acc = result["metrics"]["validation"]["accuracy"]
        tunable_values = {
            f"{comp}.{param}": result["config"][comp][param]
            for comp, param in param_names
        }
        marker = " ★" if i == 0 else ""
        print(
            f"  {i + 1}. {val_acc['mean']:.4f} ± {val_acc['std']:.4f}  {tunable_values}{marker}"
        )

    best_config = results[0]["config"]
    print(f"\nBest config saved to {exp_dir}/search_results.json")
    return best_config
