"""Per-filter L2 / mean / std analysis of trained STDP SNN encoders.

Surveys logs under ``logs/<dataset>/.../model.pth``, ranks runs by validation
accuracy (when ``metrics.json`` is present), and plots per-filter L2 / mean /
std distributions for best vs worst SNN runs alongside the ANN bimodal
classifier encoder for direct comparison.

Output: ``logs/snn_weight_analysis/snn_filter_distribution.png``
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spiking import load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS = PROJECT_ROOT / "logs"
OUT_DIR = LOGS / "snn_weight_analysis"


@dataclass
class Run:
    name: str
    model_path: Path
    val_acc: float | None
    weights_2d: torch.Tensor  # (num_filters, fan_in)


def _val_acc(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    m = json.loads(metrics_path.read_text())
    if "linear_svc" in m:
        return m["linear_svc"]["validation"]["accuracy"]
    if "validation" in m:
        return m["validation"]["accuracy"]
    return None


def _load_run(model_path: Path, name: str) -> Run | None:
    try:
        layer = load_model(str(model_path))
    except Exception as e:
        print(f"  skip {name}: {e}")
        return None
    w = layer.weights.detach().cpu()  # (N, fan_in)
    if w.dim() != 2:
        w = w.flatten(1)
    return Run(
        name=name,
        model_path=model_path,
        val_acc=_val_acc(model_path.parent / "metrics.json"),
        weights_2d=w,
    )


def _collect(pattern: str, label: str) -> list[Run]:
    runs: list[Run] = []
    for p in sorted(glob.glob(str(LOGS / pattern))):
        path = Path(p)
        rel = path.relative_to(LOGS)
        run = _load_run(path, f"{label}/{rel}")
        if run is not None:
            runs.append(run)
    return runs


def _stats(w: torch.Tensor) -> dict[str, np.ndarray]:
    return {
        "l2": w.norm(dim=1).numpy(),
        "mean": w.mean(dim=1).numpy(),
        "std": w.std(dim=1).numpy(),
    }


def _ann_classifier_weights() -> torch.Tensor | None:
    ckpt_path = (
        LOGS / "ann_bimodal_classifier/cifar10_whitened/seed_1/lambda_0.1/checkpoint.pt"
    )
    if not ckpt_path.exists():
        return None
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc = state.get("encoder", state.get("state_dict", state))
    w = enc.get("weight") if isinstance(enc, dict) else None
    if w is None:
        return None
    return w.flatten(1)


def main() -> None:
    cifar_runs = _collect("cifar10_whitened/sweep/nf_256/*/seed_*/model.pth", "cifar10")
    fmnist_runs = _collect("fashion_mnist/nf_256/*/seed_*/model.pth", "fmnist")
    print(f"loaded {len(cifar_runs)} cifar10 runs, {len(fmnist_runs)} fmnist runs")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row, (label, runs) in enumerate(
        [("CIFAR-10 whitened", cifar_runs), ("Fashion-MNIST", fmnist_runs)]
    ):
        ranked = [r for r in runs if r.val_acc is not None]
        ranked.sort(key=lambda r: r.val_acc, reverse=True)

        if not ranked:
            for col in range(3):
                axes[row, col].text(
                    0.5, 0.5, f"no metrics for {label}", ha="center", va="center"
                )
                axes[row, col].axis("off")
            continue

        top = ranked[: min(3, len(ranked))]
        bot = ranked[-min(3, len(ranked)) :] if len(ranked) > 3 else []

        for col, key in enumerate(["l2", "mean", "std"]):
            ax = axes[row, col]
            for r in top:
                vals = _stats(r.weights_2d)[key]
                ax.hist(
                    vals,
                    bins=40,
                    alpha=0.5,
                    label=f"top {r.val_acc:.3f} {Path(r.name).parts[-3]}",
                )
            for r in bot:
                vals = _stats(r.weights_2d)[key]
                ax.hist(
                    vals,
                    bins=40,
                    alpha=0.3,
                    linestyle="--",
                    label=f"bot {r.val_acc:.3f} {Path(r.name).parts[-3]}",
                    histtype="step",
                )
            ax.set_title(f"{label} | per-filter {key}")
            ax.set_xlabel(key)
            ax.set_ylabel("count")
            ax.legend(fontsize=6, loc="best")
            ax.grid(alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "snn_filter_distribution.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")

    # SNN best vs ANN classifier comparison
    ann_w = _ann_classifier_weights()
    if ann_w is None:
        print("ann classifier weights not found — skipping comparison plot")
        return

    best_cifar = max(
        (r for r in cifar_runs if r.val_acc is not None),
        key=lambda r: r.val_acc,
        default=None,
    )
    best_fmnist = max(
        (r for r in fmnist_runs if r.val_acc is not None),
        key=lambda r: r.val_acc,
        default=None,
    )
    ann_stats = _stats(ann_w)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key in zip(axes, ["l2", "mean", "std"]):
        if best_cifar:
            ax.hist(
                _stats(best_cifar.weights_2d)[key],
                bins=40,
                alpha=0.55,
                label=f"best SNN cifar10 {best_cifar.val_acc:.3f}",
            )
        if best_fmnist:
            ax.hist(
                _stats(best_fmnist.weights_2d)[key],
                bins=40,
                alpha=0.55,
                label=f"best SNN fmnist {best_fmnist.val_acc:.3f}",
            )
        ax.hist(ann_stats[key], bins=40, alpha=0.55, label="ANN classifier λ=0.1")
        ax.set_title(f"per-filter {key}")
        ax.set_xlabel(key)
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out2 = OUT_DIR / "snn_vs_ann_filter_distribution.png"
    fig.savefig(out2, dpi=120)
    plt.close(fig)
    print(f"saved {out2}")

    # Summary table
    summary = {}
    for label, run in [
        ("ANN_classifier", None),
        ("best_cifar_snn", best_cifar),
        ("best_fmnist_snn", best_fmnist),
    ]:
        if label == "ANN_classifier":
            s = _stats(ann_w)
        elif run is None:
            continue
        else:
            s = _stats(run.weights_2d)
        summary[label] = {
            "val_acc": run.val_acc if run else None,
            "fan_in": int(
                ann_w.shape[1] if label == "ANN_classifier" else run.weights_2d.shape[1]
            ),
            "l2_mean": float(s["l2"].mean()),
            "l2_std": float(s["l2"].std()),
            "mean_mean": float(s["mean"].mean()),
            "mean_std": float(s["mean"].std()),
            "std_mean": float(s["std"].mean()),
            "std_std": float(s["std"].std()),
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
