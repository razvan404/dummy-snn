"""Seed-1 FMnist whisker plot: train + val accuracy per ordering.

Layout mirrors the CIFAR svc_ordering_whiskers.png: baseline box (1 point for
seed 1 only), then 5 ordering boxes. 'random' pools 3 ordering seeds (3 points).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "logs/fashion_mnist/nf_256/tobj_0.85/seed_{seed}/greedy_opt_svc/{ord}/results.json"
snn_seeds = [1]

groups = [
    ("oldest_winner", "oldest\nwinner", ["oldest_winner"]),
    ("recent_winner", "recent\nwinner", ["recent_winner"]),
    ("random", "random\n(3 order seeds)", ["random_s1", "random_s2", "random_s3"]),
    ("ascending_importance", "asc\nimportance", ["ascending_importance"]),
    ("descending_importance", "desc\nimportance", ["descending_importance"]),
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
    r = json.load(open(BASE.format(seed=s, ord="oldest_winner")))
    base_train.append(r["baseline"]["train_acc"] * 100)
    base_val.append(r["baseline"]["val_acc"] * 100)

labels = [f"baseline\n({len(snn_seeds)} SNN seed)"] + [g[1] for g in groups]
train_data = [base_train] + [train_vals[g[0]] for g in groups]
val_data = [base_val] + [val_vals[g[0]] for g in groups]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
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
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

box(axes[0], train_data, "Train accuracy across orderings (FMnist seed 1)", "Train accuracy (%)")
box(axes[1], val_data, "Validation accuracy across orderings (FMnist seed 1)", "Val accuracy (%)")

plt.suptitle(
    "SVC-targeted greedy threshold optimization — FMnist (t_obj=0.85, seed 1)\n"
    f"(each point = one ordering run; random pools 3 points; other orderings 1 point)",
    fontsize=12, y=1.03,
)
plt.tight_layout()
out = "/bigdata/md5_t81ab4c87d31f439ff161a3ad27d/dummy-snn/fmnist_seed1_whiskers.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved {out}")

# Also print a summary table
print("\nSeed 1 summary:")
print(f"  baseline: train {base_train[0]:.2f}  val {base_val[0]:.2f}")
for key, lbl, _ in groups:
    tr = train_vals[key]; va = val_vals[key]
    if not tr: continue
    tr_str = f"{np.mean(tr):.2f}" + (f" (n={len(tr)})" if len(tr) > 1 else "")
    va_str = f"{np.mean(va):.2f}" + (f" (n={len(va)})" if len(va) > 1 else "")
    dt = np.mean(tr) - base_train[0]
    dv = np.mean(va) - base_val[0]
    print(f"  {key:<22s}  train {tr_str:<12s}  val {va_str:<12s}  Δ train {dt:+.2f}  Δ val {dv:+.2f}")
