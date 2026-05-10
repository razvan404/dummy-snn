"""Combined CIFAR + FMnist whisker plot, stacked vertically.

2x2 grid: top row = CIFAR (train, val), bottom row = FMnist (train, val).
Each dataset gets a bold row title via suptitle-per-row using row-wise figtext.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CIFAR_BASE = "logs/cifar10_whitened/sweep/nf_256/tobj_0.70/seed_{seed}/greedy_opt_svc/{ord}/results.json"
FMN_BASE = "logs/fashion_mnist/nf_256/tobj_0.85/seed_{seed}/greedy_opt_svc/{ord}/results.json"

cifar_seeds = [1, 2, 3, 4, 5, 6, 7, 8]
fmn_seeds = []
for s in [1, 2, 3, 4, 5, 6, 7, 8]:
    if any(Path(FMN_BASE.format(seed=s, ord=o)).exists()
           for o in ["oldest_winner", "recent_winner", "random_s1"]):
        fmn_seeds.append(s)

groups = [
    ("oldest_winner", "oldest\nwinner", ["oldest_winner"]),
    ("recent_winner", "recent\nwinner", ["recent_winner"]),
    ("ascending_importance", "asc\nimpt.", ["ascending_importance"]),
    ("descending_importance", "desc\nimpt.", ["descending_importance"]),
    ("random", "random", ["random_s1", "random_s2", "random_s3"]),
]


def collect(base, seeds):
    train_vals, val_vals = {}, {}
    for group_key, _, subs in groups:
        train_vals[group_key] = []
        val_vals[group_key] = []
        for s in seeds:
            for sub in subs:
                p = Path(base.format(seed=s, ord=sub))
                if not p.exists():
                    continue
                r = json.load(open(p))
                train_vals[group_key].append(r["final"]["train_acc"] * 100)
                val_vals[group_key].append(r["final"]["val_acc"] * 100)
    base_train, base_val = [], []
    for s in seeds:
        for o in ["oldest_winner", "recent_winner", "random_s1", "random_s2",
                  "random_s3", "ascending_importance", "descending_importance"]:
            p = Path(base.format(seed=s, ord=o))
            if p.exists():
                r = json.load(open(p))
                base_train.append(r["baseline"]["train_acc"] * 100)
                base_val.append(r["baseline"]["val_acc"] * 100)
                break
    labels = ["baseline"] + [g[1] for g in groups]
    train_data = [base_train] + [train_vals[g[0]] for g in groups]
    val_data = [base_val] + [val_vals[g[0]] for g in groups]
    return labels, train_data, val_data


cifar_labels, cifar_tr, cifar_va = collect(CIFAR_BASE, cifar_seeds)
fmn_labels, fmn_tr, fmn_va = collect(FMN_BASE, fmn_seeds)

BASELINE_FACE, BASELINE_EDGE = "#d9d9d9", "#707070"
OPT_FACE, OPT_EDGE = "#9db8d0", "#37557b"


def box(ax, data, labels, title, ylabel):
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
    )
    for i, patch in enumerate(bp["boxes"]):
        if i == 0:
            patch.set_facecolor(BASELINE_FACE)
            patch.set_edgecolor(BASELINE_EDGE)
        else:
            patch.set_facecolor(OPT_FACE)
            patch.set_edgecolor(OPT_EDGE)
    for i, d in enumerate(data, start=1):
        color = "#4a4a4a" if i == 1 else "#1f3c5a"
        ax.scatter([i] * len(d), d, color=color, s=22, zorder=3)
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.3, axis="y")


fig = plt.figure(figsize=(12, 4.8), constrained_layout=True)
subfigs = fig.subfigures(2, 1, hspace=0.05)

from matplotlib.ticker import MaxNLocator, FormatStrFormatter

subfigs[0].suptitle("Fashion-MNIST accuracy (%)", fontsize=18)
fmn_axes = subfigs[0].subplots(1, 2)
box(fmn_axes[0], fmn_tr, fmn_labels, "Train", "")
box(fmn_axes[1], fmn_va, fmn_labels, "Validation", "")

subfigs[1].suptitle("CIFAR-10 accuracy (%)", fontsize=18)
cifar_axes = subfigs[1].subplots(1, 2)
box(cifar_axes[0], cifar_tr, cifar_labels, "Train", "")
box(cifar_axes[1], cifar_va, cifar_labels, "Validation", "")
for ax in (*fmn_axes, *cifar_axes):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
out = "/bigdata/md5_t81ab4c87d31f439ff161a3ad27d/dummy-snn/combined_whiskers.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
