"""FMnist whisker plot: train + val accuracy per ordering, across available SNN seeds.

Same layout as the CIFAR svc_ordering_whiskers.png. Skips orderings/seeds
that don't yet have results.json (pipeline still running).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "logs/fashion_mnist/nf_256/tobj_0.85/seed_{seed}/greedy_opt_svc/{ord}/results.json"
# Discover which seeds have at least one result
snn_seeds = []
for s in [1, 2, 3, 4, 5]:
    if any(Path(BASE.format(seed=s, ord=o)).exists()
           for o in ["oldest_winner", "recent_winner", "random_s1"]):
        snn_seeds.append(s)
print(f"Seeds with results: {snn_seeds}")

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
            p = Path(BASE.format(seed=s, ord=sub))
            if not p.exists(): continue
            r = json.load(open(p))
            train_vals[group_key].append(r["final"]["train_acc"] * 100)
            val_vals[group_key].append(r["final"]["val_acc"] * 100)

base_train, base_val = [], []
for s in snn_seeds:
    # any one ordering has the baseline for that seed
    for o in ["oldest_winner", "recent_winner", "random_s1", "random_s2", "random_s3",
              "ascending_importance", "descending_importance"]:
        p = Path(BASE.format(seed=s, ord=o))
        if p.exists():
            r = json.load(open(p))
            base_train.append(r["baseline"]["train_acc"] * 100)
            base_val.append(r["baseline"]["val_acc"] * 100)
            break

n = len(snn_seeds)
labels = ["baseline"] + [g[1] for g in groups]
train_data = [base_train] + [train_vals[g[0]] for g in groups]
val_data = [base_val] + [val_vals[g[0]] for g in groups]

fig, axes = plt.subplots(1, 2, figsize=(12, 2.1), constrained_layout=True)
BASELINE_FACE, BASELINE_EDGE = "#d9d9d9", "#707070"
OPT_FACE, OPT_EDGE = "#9db8d0", "#37557b"

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
            patch.set_facecolor(BASELINE_FACE); patch.set_edgecolor(BASELINE_EDGE)
        else:
            patch.set_facecolor(OPT_FACE); patch.set_edgecolor(OPT_EDGE)
    for i, d in enumerate(data, start=1):
        color = "#4a4a4a" if i == 1 else "#1f3c5a"
        ax.scatter([i] * len(d), d, color=color, s=30, zorder=3)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3, axis="y")

box(axes[0], train_data, "Train", "Train accuracy (%)")
box(axes[1], val_data, "Validation", "Val accuracy (%)")

fig.suptitle("FashionMNIST results", fontsize=13)
out = "/bigdata/md5_t81ab4c87d31f439ff161a3ad27d/dummy-snn/fmnist_whiskers.png"
plt.savefig(out, dpi=100, bbox_inches="tight")
print(f"Saved {out}")

# Summary table across available seeds
print(f"\nFMnist summary ({n} SNN seed{'s' if n > 1 else ''}):")
print(f"  baseline: train {np.mean(base_train):.2f}  val {np.mean(base_val):.2f}")
for key, lbl, _ in groups:
    tr = train_vals[key]; va = val_vals[key]
    if not tr: continue
    dt = np.mean(tr) - np.mean(base_train)
    dv = np.mean(va) - np.mean(base_val)
    print(f"  {key:<22s}  train {np.mean(tr):.2f} (n={len(tr)})  val {np.mean(va):.2f} (n={len(va)})  Δ train {dt:+.2f}  Δ val {dv:+.2f}")
