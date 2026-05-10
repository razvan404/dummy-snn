"""Whisker/box plot of final train and val accuracy per ordering.

The baseline (unoptimized thresholds) is shown as its own leftmost box
(3 points = 3 SNN seeds). Each ordering is one box pooled across all SNN
seeds; 'random' additionally pools 3 ordering seeds (9 points total).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_FMT = "logs/cifar10_whitened/sweep/nf_256/tobj_0.70/seed_{seed}/greedy_opt_svc/{ord}/results.json"
snn_seeds = [1, 2, 3, 4, 5]

groups = [
    ("oldest_winner", "oldest\nwinner", ["oldest_winner"]),
    ("recent_winner", "recent\nwinner", ["recent_winner"]),
    ("ascending_importance", "asc\nimportance", ["ascending_importance"]),
    ("descending_importance", "desc\nimportance", ["descending_importance"]),
    ("random", "random", ["random_s1", "random_s2", "random_s3"]),
]

train_vals, val_vals = {}, {}
for group_key, _, subs in groups:
    train_vals[group_key] = []
    val_vals[group_key] = []
    for s in snn_seeds:
        for sub in subs:
            r = json.load(open(BASE_FMT.format(seed=s, ord=sub)))
            train_vals[group_key].append(r["final"]["train_acc"] * 100)
            val_vals[group_key].append(r["final"]["val_acc"] * 100)

# Baselines — one per SNN seed
base_train, base_val = [], []
for s in snn_seeds:
    r = json.load(open(BASE_FMT.format(seed=s, ord="oldest_winner")))
    base_train.append(r["baseline"]["train_acc"] * 100)
    base_val.append(r["baseline"]["val_acc"] * 100)

labels = ["baseline"] + [g[1] for g in groups]
train_data = [base_train] + [train_vals[g[0]] for g in groups]
val_data = [base_val] + [val_vals[g[0]] for g in groups]

fig, axes = plt.subplots(1, 2, figsize=(12, 2.1), constrained_layout=True)

BASELINE_FACE = "#d9d9d9"
BASELINE_EDGE = "#707070"
OPT_FACE = "#9db8d0"
OPT_EDGE = "#37557b"


def box(ax, data, title, ylabel):
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
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3, axis="y")


box(axes[0], train_data, "Train", "Train accuracy (%)")
box(axes[1], val_data, "Validation", "Val accuracy (%)")

fig.suptitle("CIFAR-10 results", fontsize=13)
out = "/bigdata/md5_t81ab4c87d31f439ff161a3ad27d/dummy-snn/svc_ordering_whiskers.png"
plt.savefig(out, dpi=100, bbox_inches="tight")
print(f"Saved {out}")
