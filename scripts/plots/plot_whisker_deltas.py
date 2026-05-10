"""Whisker plot of Δ (ordering − baseline) per seed, for seeds 1–5.

Same visual format as combined_whiskers.png: 2×2 layout with FMNIST on top,
CIFAR on bottom, Train (left) and Validation (right). Each subplot has one
box per ordering (no baseline box — Δ is relative to baseline by definition).
Zero line drawn for reference.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

CIFAR_BASE = "logs/cifar10_whitened/sweep/nf_256/tobj_0.70/seed_{seed}/greedy_opt_svc/{ord}/results.json"
FMN_BASE = "logs/fashion_mnist/nf_256/tobj_0.85/seed_{seed}/greedy_opt_svc/{ord}/results.json"

SEEDS = [1, 2, 3, 4, 5]

groups = [
    ("oldest_winner", "oldest\nwinner", ["oldest_winner"]),
    ("recent_winner", "recent\nwinner", ["recent_winner"]),
    ("ascending_importance", "asc\nimpt.", ["ascending_importance"]),
    ("descending_importance", "desc\nimpt.", ["descending_importance"]),
    ("random", "random", ["random_s1", "random_s2", "random_s3"]),
]


def collect_deltas(base):
    """For each ordering, return (train_deltas, val_deltas) pooled across seeds.

    Also returns per-seed deltas (for connecting lines): for random, the mean
    over the 3 ordering seeds is used as the seed's single representative.
    """
    train_vals, val_vals = {}, {}
    per_seed_tr, per_seed_va = {s: [] for s in SEEDS}, {s: [] for s in SEEDS}
    for group_key, _, subs in groups:
        train_vals[group_key] = []
        val_vals[group_key] = []
    for s in SEEDS:
        ref_path = None
        for o in ["oldest_winner", "recent_winner", "random_s1", "random_s2",
                  "random_s3", "ascending_importance", "descending_importance"]:
            p = Path(base.format(seed=s, ord=o))
            if p.exists():
                ref_path = p
                break
        if ref_path is None:
            per_seed_tr[s] = [None] * len(groups)
            per_seed_va[s] = [None] * len(groups)
            continue
        ref = json.loads(ref_path.read_text())
        btr = ref["baseline"]["train_acc"] * 100
        bva = ref["baseline"]["val_acc"] * 100
        for group_key, _, subs in groups:
            seed_tr_vals, seed_va_vals = [], []
            for sub in subs:
                p = Path(base.format(seed=s, ord=sub))
                if not p.exists():
                    continue
                r = json.loads(p.read_text())
                seed_tr_vals.append(r["final"]["train_acc"] * 100 - btr)
                seed_va_vals.append(r["final"]["val_acc"] * 100 - bva)
            train_vals[group_key].extend(seed_tr_vals)
            val_vals[group_key].extend(seed_va_vals)
            per_seed_tr[s].append(sum(seed_tr_vals) / len(seed_tr_vals) if seed_tr_vals else None)
            per_seed_va[s].append(sum(seed_va_vals) / len(seed_va_vals) if seed_va_vals else None)
    labels = [g[1] for g in groups]
    train_data = [train_vals[g[0]] for g in groups]
    val_data = [val_vals[g[0]] for g in groups]
    return labels, train_data, val_data, per_seed_tr, per_seed_va


fmn_labels, fmn_tr, fmn_va, _, _ = collect_deltas(FMN_BASE)
cif_labels, cif_tr, cif_va, _, _ = collect_deltas(CIFAR_BASE)

OPT_FACE, OPT_EDGE = "#9db8d0", "#37557b"


def box(ax, data, labels, title):
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(OPT_FACE)
        patch.set_edgecolor(OPT_EDGE)
    for i, d in enumerate(data, start=1):
        ax.scatter([i] * len(d), d, color="#1f3c5a", s=22, zorder=3)
    ax.axhline(0, color="#707070", linewidth=1, linestyle="--", alpha=0.7, zorder=1)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.3, axis="y")


fig = plt.figure(figsize=(12, 4.8), constrained_layout=True)
subfigs = fig.subfigures(2, 1, hspace=0.05)

subfigs[0].suptitle("Fashion-MNIST Δ accuracy (pp)", fontsize=18)
fmn_axes = subfigs[0].subplots(1, 2)
box(fmn_axes[0], fmn_tr, fmn_labels, "Train")
box(fmn_axes[1], fmn_va, fmn_labels, "Validation")

subfigs[1].suptitle("CIFAR-10 Δ accuracy (pp)", fontsize=18)
cif_axes = subfigs[1].subplots(1, 2)
box(cif_axes[0], cif_tr, cif_labels, "Train")
box(cif_axes[1], cif_va, cif_labels, "Validation")

for ax in (*fmn_axes, *cif_axes):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

out = "/bigdata/md5_t81ab4c87d31f439ff161a3ad27d/dummy-snn/combined_delta_whiskers.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
