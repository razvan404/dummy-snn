"""Baseline accuracy (x) vs post-threshold-update accuracy (y), per ordering.

Shows consistency of improvement at the run level: points above the y=x
diagonal improve, below it regress. Seeds 1–5, both datasets, Train + Val.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

CIFAR_BASE = "logs/cifar10_whitened/sweep/nf_256/tobj_0.70/seed_{seed}/greedy_opt_svc/{ord}/results.json"
FMN_BASE = "logs/fashion_mnist/nf_256/tobj_0.85/seed_{seed}/greedy_opt_svc/{ord}/results.json"

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]

# (label, subdirs, color, marker)
ORDERINGS = [
    ("oldest winner", ["oldest_winner"], "#d73027", "o"),
    ("recent winner", ["recent_winner"], "#fc8d59", "s"),
    ("asc impt.", ["ascending_importance"], "#4575b4", "^"),
    ("desc impt.", ["descending_importance"], "#5aae61", "v"),
    ("random", ["random_s1", "random_s2", "random_s3"], "#9e3dbf", "D"),
]


def collect(base):
    """Return dict: ordering_label -> (train_xs, train_ys, val_xs, val_ys)."""
    out = {}
    for name, subs, *_ in ORDERINGS:
        tx, ty, vx, vy = [], [], [], []
        for s in SEEDS:
            for sub in subs:
                p = Path(base.format(seed=s, ord=sub))
                if not p.exists():
                    continue
                r = json.loads(p.read_text())
                tx.append(r["baseline"]["train_acc"] * 100)
                ty.append(r["final"]["train_acc"] * 100)
                vx.append(r["baseline"]["val_acc"] * 100)
                vy.append(r["final"]["val_acc"] * 100)
        out[name] = (tx, ty, vx, vy)
    return out


fmn = collect(FMN_BASE)
cif = collect(CIFAR_BASE)


def scatter(ax, data, coord_tr_or_va, title, show_ylabel=True, show_xlabel=False):
    """coord_tr_or_va: 'tr' for (tx, ty), 'va' for (vx, vy)."""
    all_x, all_y = [], []
    for name, subs, color, marker in ORDERINGS:
        tx, ty, vx, vy = data[name]
        x, y = (tx, ty) if coord_tr_or_va == "tr" else (vx, vy)
        z = 2 if name == "random" else 3
        ax.scatter(x, y, color=color, marker=marker, s=70, alpha=0.85,
                   edgecolor="black", linewidth=0.6, label=name, zorder=z)
        all_x.extend(x); all_y.extend(y)

    lo = min(all_x + all_y)
    hi = max(all_x + all_y)
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color="#707070", linewidth=1.2,
            linestyle="--", alpha=0.8, zorder=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_title(title, fontsize=16)
    if show_xlabel:
        ax.set_xlabel("Baseline accuracy (%)", fontsize=15)
    if show_ylabel:
        ax.set_ylabel("Post-hoc accuracy (%)", fontsize=15)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


fig = plt.figure(figsize=(14, 3.3), constrained_layout=True)
subfigs = fig.subfigures(1, 2, wspace=0.04)

subfigs[0].suptitle("Fashion-MNIST", fontsize=18)
fmn_axes = subfigs[0].subplots(1, 2)
scatter(fmn_axes[0], fmn, "tr", "Train", show_ylabel=True)
scatter(fmn_axes[1], fmn, "va", "Validation", show_ylabel=False)

subfigs[1].suptitle("CIFAR-10", fontsize=18)
cif_axes = subfigs[1].subplots(1, 2)
scatter(cif_axes[0], cif, "tr", "Train", show_ylabel=False)
scatter(cif_axes[1], cif, "va", "Validation", show_ylabel=False)

# Shared x-label centered across the four plots
fig.supxlabel("Baseline accuracy (%)", fontsize=15)

# single shared legend at bottom
handles, labels = fmn_axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=15,
           frameon=True, bbox_to_anchor=(0.5, -0.13))

out = "/bigdata/md5_t81ab4c87d31f439ff161a3ad27d/dummy-snn/baseline_vs_optimized.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
