import argparse
import json
import re
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SERIES_COLORS = {
    "baseline": "#1f77b4",
    "pbtr": "#ff7f0e",
    "pbtr_sign": "#2ca02c",
    "random_thresh": "#d62728",
    "uniform_thresh": "#9467bd",
}


def _extract_numeric(dirname: str) -> float:
    """Extract the numeric value from a variant directory name like 'thresh_39.2'."""
    match = re.search(r"[\d.]+", dirname)
    if match is None:
        raise ValueError(f"Cannot extract numeric value from '{dirname}'")
    return float(match.group())


def _load_accuracies(path: str, split: str) -> list[float] | None:
    """Load accuracy list from a merged_results.json file, or None if missing."""
    try:
        with open(path) as f:
            data = json.load(f)
        return data[split]["accuracy"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def _collect_pbtr_accuracies(variant_dir: Path, subdir: str, split: str) -> list[float]:
    """Collect accuracies across all seed_*/subdir/merged_results.json files."""
    accuracies = []
    for path in sorted(variant_dir.glob(f"seed_*/{subdir}/merged_results.json")):
        accs = _load_accuracies(str(path), split)
        if accs is not None:
            accuracies.extend(accs)
    return accuracies


def load_comparison_data(
    base_dir: Path, split: str
) -> list[tuple[float, dict[str, list[float]]]]:
    """Load baseline, PBTR, and random threshold accuracies for all variants under base_dir.

    Returns a sorted list of (param_value, {series_name: [accuracies]}).
    Only includes series that have data.
    """
    variants = []
    for variant_path in sorted(base_dir.iterdir()):
        if not variant_path.is_dir():
            continue
        try:
            param_value = _extract_numeric(variant_path.name)
        except ValueError:
            continue

        baseline = _load_accuracies(str(variant_path / "merged_results.json"), split)
        if baseline is None:
            continue

        series = {"baseline": baseline}

        pbtr = _collect_pbtr_accuracies(variant_path, "pbtr", split)
        if pbtr:
            series["pbtr"] = pbtr

        pbtr_sign = _collect_pbtr_accuracies(variant_path, "pbtr_sign", split)
        if pbtr_sign:
            series["pbtr_sign"] = pbtr_sign

        random_thresh = _collect_pbtr_accuracies(variant_path, "random_thresh", split)
        if random_thresh:
            series["random_thresh"] = random_thresh

        uniform_thresh = _collect_pbtr_accuracies(variant_path, "uniform_thresh", split)
        if uniform_thresh:
            series["uniform_thresh"] = uniform_thresh

        variants.append((param_value, series))

    variants.sort(key=lambda x: x[0])
    return variants


def plot_comparison(
    variants: list[tuple[float, dict[str, list[float]]]],
    output_path: str,
    *,
    network_type: str,
    split: str,
):
    """Create a grouped boxplot comparing baseline vs PBTR variants."""
    if not variants:
        print("No data to plot.")
        return

    all_series = []
    for _val, series in variants:
        for name in series:
            if name not in all_series:
                all_series.append(name)

    n_series = len(all_series)
    box_width = 0.6 / n_series
    fig, ax = plt.subplots(figsize=(max(8, len(variants) * 1.2), 5))

    positions_map = {}
    tick_positions = []
    tick_labels = []

    for group_idx, (param_value, series) in enumerate(variants):
        center = group_idx
        tick_positions.append(center)
        tick_labels.append(str(param_value))

        for series_idx, name in enumerate(all_series):
            if name not in series:
                continue
            offset = (series_idx - (n_series - 1) / 2) * box_width
            pos = center + offset
            bp = ax.boxplot(
                [series[name]],
                positions=[pos],
                widths=box_width * 0.85,
                patch_artist=True,
                manage_ticks=False,
            )
            color = SERIES_COLORS.get(name, "#999999")
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            for median in bp["medians"]:
                median.set_color("black")
            positions_map[name] = color

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(f"{split} accuracy")
    ax.set_xlabel("parameter value")
    ax.set_title(f"{network_type}: post-training comparison ({split} accuracy)")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7, label=name)
        for name, color in positions_map.items()
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot baseline vs PBTR comparison boxplots"
    )
    parser.add_argument("dataset", type=str)
    parser.add_argument("--network-type", type=str, required=True)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--split", type=str, default="validation")
    args = parser.parse_args()

    base_dir = Path(
        f"logs/{args.dataset}/layer_{args.layer_idx + 1}/{args.network_type}"
    )
    if not base_dir.is_dir():
        print(f"Directory not found: {base_dir}")
        return

    variants = load_comparison_data(base_dir, args.split)
    output_path = str(base_dir / "comparison.png")
    plot_comparison(
        variants, output_path, network_type=args.network_type, split=args.split
    )


if __name__ == "__main__":
    main()
