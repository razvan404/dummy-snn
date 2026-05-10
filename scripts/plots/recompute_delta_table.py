"""Recompute the Δ-accuracy LaTeX table from all available results.json files.

Auto-discovers seeds for both datasets. Δ = final - baseline (pp). Random pools
all (seed × ordering_seed) runs into a single sample. mean ± sample-std (ddof=1).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FMN = Path("logs/fashion_mnist/nf_256/tobj_0.85")
CIF = Path("logs/cifar10_whitened/sweep/nf_256/tobj_0.70")

ORDERINGS = [
    ("Oldest-winner", ["oldest_winner"]),
    ("Recent-winner", ["recent_winner"]),
    ("Asc.\\ importance", ["ascending_importance"]),
    ("Desc.\\ importance", ["descending_importance"]),
    ("Random", ["random_s1", "random_s2", "random_s3"]),
]

ORDER_DIRS = [
    "oldest_winner", "recent_winner", "random_s1", "random_s2",
    "random_s3", "ascending_importance", "descending_importance",
]


def discover_seeds(base: Path) -> list[int]:
    """Return sorted seeds that have at least one ordering results.json."""
    seeds: list[int] = []
    for d in base.glob("seed_*"):
        s = int(d.name.removeprefix("seed_"))
        if any((d / "greedy_opt_svc" / o / "results.json").exists() for o in ORDER_DIRS):
            seeds.append(s)
    return sorted(seeds)


def baseline(base: Path, seed: int) -> tuple[float, float] | None:
    """Return (baseline_train, baseline_val) in pp from any completed ordering."""
    for o in ORDER_DIRS:
        p = base / f"seed_{seed}" / "greedy_opt_svc" / o / "results.json"
        if p.exists():
            r = json.loads(p.read_text())
            return r["baseline"]["train_acc"] * 100, r["baseline"]["val_acc"] * 100
    return None


def deltas(base: Path, seeds: list[int], subs: list[str]) -> tuple[list[float], list[float]]:
    """Per-run train and val Δ (pp) for all (seed × sub) with a results.json."""
    tr, va = [], []
    for s in seeds:
        b = baseline(base, s)
        if b is None:
            continue
        btr, bva = b
        for sub in subs:
            p = base / f"seed_{s}" / "greedy_opt_svc" / sub / "results.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            tr.append(r["final"]["train_acc"] * 100 - btr)
            va.append(r["final"]["val_acc"] * 100 - bva)
    return tr, va


def fmt(x: list[float]) -> str:
    if not x:
        return "--"
    m, s = float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    sign = "+" if m >= 0 else ""
    return f"${sign}{m:.2f}\\!\\pm\\!{s:.2f}$"


def main() -> None:
    fmn_seeds = discover_seeds(FMN)
    cif_seeds = discover_seeds(CIF)
    print(f"% FMNIST seeds: {fmn_seeds}")
    print(f"% CIFAR  seeds: {cif_seeds}")
    print()

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Accuracy improvement $\Delta$ (percentage) over the baseline.}")
    print(r"\label{tab:deltas}")
    print(r"\renewcommand{\arraystretch}{1.15}")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"\textbf{Ordering} &")
    print(r"\makecell{\textbf{FMNIST}\\\textbf{Train}} &")
    print(r"\makecell{\textbf{FMNIST}\\\textbf{Val}} &")
    print(r"\makecell{\textbf{CIFAR}\\\textbf{Train}} &")
    print(r"\makecell{\textbf{CIFAR}\\\textbf{Val}}\\")
    print(r"\hline")
    for name, subs in ORDERINGS:
        ftr, fva = deltas(FMN, fmn_seeds, subs)
        ctr, cva = deltas(CIF, cif_seeds, subs)
        print(f"{name:<18s} & {fmt(ftr)} & {fmt(fva)} & {fmt(ctr)} & {fmt(cva)} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print()
    print("% Sample counts (train / val):")
    for name, subs in ORDERINGS:
        ftr, _ = deltas(FMN, fmn_seeds, subs)
        ctr, _ = deltas(CIF, cif_seeds, subs)
        print(f"%   {name:<18s}  FMN n={len(ftr)}  CIF n={len(ctr)}")


if __name__ == "__main__":
    main()
