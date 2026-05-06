"""Box/whisker plot comparing ordering methods across SNN seeds.

X-axis: baseline, descending_imp, ascending_imp, recent_winner, oldest_winner, random
Y-axis: absolute val accuracy (%)
Each box aggregates across SNN seeds; random also pools random shuffle seeds.
"""

import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

ORDERING_GROUPS = {
    "descending_importance": "descending\nimportance",
    "ascending_importance": "ascending\nimportance",
    "recent_winner": "recent\nwinner",
    "oldest_winner": "oldest\nwinner",
}
RANDOM_PREFIX = "random"


def main() -> None:
    parser = argparse.ArgumentParser(description="Box plot of ordering comparison")
    parser.add_argument(
        "--comparison-jsons",
        nargs="+",
        required=True,
        help="comparison.json files from compare_orderings (one per SNN seed)",
    )
    parser.add_argument("--output", default="ordering_comparison.png")
    parser.add_argument("--dataset", default="CIFAR-10")
    args = parser.parse_args()

    method_names = [
        "baseline",
        "descending\nimportance",
        "ascending\nimportance",
        "recent\nwinner",
        "oldest\nwinner",
        "random",
    ]
    ridge_vals = {m: [] for m in method_names}
    svc_vals = {m: [] for m in method_names}

    for json_path in args.comparison_jsons:
        with open(json_path) as f:
            table = json.load(f)

        for row in table:
            ordering = row["ordering"]
            ridge_v = row["ridge_val"] * 100
            svc_v = row["svc_val"] * 100

            if ordering == "baseline":
                group = "baseline"
            elif ordering in ORDERING_GROUPS:
                group = ORDERING_GROUPS[ordering]
            elif ordering.startswith(RANDOM_PREFIX):
                group = "random"
            else:
                continue

            ridge_vals[group].append(ridge_v)
            svc_vals[group].append(svc_v)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#888888", "#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66"]

    for ax, data_dict, title in [
        (axes[0], ridge_vals, f"{args.dataset} — Ridge Val Accuracy"),
        (axes[1], svc_vals, f"{args.dataset} — SVC Val Accuracy"),
    ]:
        data = [data_dict[m] for m in method_names]
        if not any(len(d) > 0 for d in data):
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        bp = ax.boxplot(
            data,
            labels=method_names,
            patch_artist=True,
            widths=0.6,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="black", markersize=5),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        # Scatter individual points
        rng = np.random.RandomState(42)
        for i, d in enumerate(data):
            x = rng.normal(i + 1, 0.06, size=len(d))
            ax.scatter(x, d, alpha=0.6, s=25, color="black", zorder=3)

        # Draw baseline mean as horizontal reference
        if ridge_vals["baseline"]:
            baseline_mean = np.mean(data_dict["baseline"])
            ax.axhline(
                baseline_mean,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                label=f"baseline mean: {baseline_mean:.2f}%",
            )
            ax.legend(fontsize=8)

        ax.set_ylabel("Val accuracy (%)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
