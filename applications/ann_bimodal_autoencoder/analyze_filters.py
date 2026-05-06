"""Per-filter analysis for a trained bimodal autoencoder run.

Usage::

    python -m applications.ann_bimodal_autoencoder.analyze_filters \\
        --run-dir logs/ann_bimodal_autoencoder/cifar10/seed_1/lambda_0.1
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_encoder_weights(run_dir: Path) -> torch.Tensor:
    ckpt = torch.load(run_dir / "checkpoint.pt", weights_only=True, map_location="cpu")
    # encoder.state_dict() contains the single key "weight"
    return ckpt["encoder"]["weight"]


def _save_l2_summary(weights: torch.Tensor, path: Path) -> None:
    w = weights.detach().cpu().numpy()  # (nf, C, kH, kW)
    nf = w.shape[0]
    n_per_filter = int(np.prod(w.shape[1:]))

    sum_w = w.reshape(nf, -1).sum(axis=1)
    mean_w = sum_w / n_per_filter
    l2 = np.sqrt((w**2).sum(axis=(1, 2, 3)))
    rms = l2 / np.sqrt(n_per_filter)
    near0 = (w <= 0.1).reshape(nf, -1).mean(axis=1)
    near1 = (w >= 0.9).reshape(nf, -1).mean(axis=1)
    midband = 1.0 - near0 - near1

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].hist(l2, bins=40, color="C0", edgecolor="black", linewidth=0.3)
    axes[0, 0].set_xlabel("filter L2 norm")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].set_title(f"L2 norm  (n={nf})")

    axes[0, 1].hist(mean_w, bins=40, color="C2", edgecolor="black", linewidth=0.3)
    axes[0, 1].axvline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7)
    axes[0, 1].set_xlabel("mean weight per filter (sum / N)")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].set_title("Mean weight per filter")

    axes[0, 2].hist(rms, bins=40, color="C4", edgecolor="black", linewidth=0.3)
    axes[0, 2].set_xlabel("RMS per filter (L2 / √N)")
    axes[0, 2].set_ylabel("count")
    axes[0, 2].set_title("Per-weight RMS (length-normalized)")

    order = np.argsort(l2)[::-1]
    axes[1, 0].bar(
        np.arange(nf), l2[order], color="C0", edgecolor="black", linewidth=0.2
    )
    axes[1, 0].set_xlabel("filter rank (sorted by L2)")
    axes[1, 0].set_ylabel("L2 norm")
    axes[1, 0].set_title("Sorted L2 norms")

    order_mean = np.argsort(mean_w)[::-1]
    axes[1, 1].bar(
        np.arange(nf), mean_w[order_mean], color="C2", edgecolor="black", linewidth=0.2
    )
    axes[1, 1].axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7)
    axes[1, 1].set_xlabel("filter rank (sorted by mean)")
    axes[1, 1].set_ylabel("mean weight")
    axes[1, 1].set_title("Sorted mean weights")

    axes[1, 2].bar(np.arange(nf), near1[order], color="C1", label="≥0.9")
    axes[1, 2].bar(
        np.arange(nf), near0[order], bottom=near1[order], color="C0", label="≤0.1"
    )
    axes[1, 2].bar(
        np.arange(nf),
        midband[order],
        bottom=near0[order] + near1[order],
        color="C7",
        label="middle",
    )
    axes[1, 2].set_xlabel("filter rank (sorted by L2)")
    axes[1, 2].set_ylabel("fraction of weights")
    axes[1, 2].set_title("Per-filter weight composition")
    axes[1, 2].legend(loc="lower right")
    axes[1, 2].set_ylim(0, 1)

    fig.suptitle(f"Per-filter analysis — encoder shape {tuple(weights.shape)}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return l2, order, near0, near1


def _save_filter_grid(
    weights: torch.Tensor,
    indices: np.ndarray,
    path: Path,
    title: str,
    ncols: int = 8,
    rescale_per_filter: bool = True,
) -> None:
    """Render a selected subset of filters as RGB images."""
    w = weights.detach().cpu().numpy()
    n = len(indices)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.4, nrows * 1.4))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    for slot in range(nrows * ncols):
        ax = axes[slot // ncols, slot % ncols]
        ax.axis("off")
        if slot >= n:
            continue
        idx = indices[slot]
        filt = w[idx]
        if filt.shape[0] == 3:
            rgb = filt.transpose(1, 2, 0)
        elif filt.shape[0] == 1:
            rgb = np.stack([filt[0]] * 3, axis=-1)
        else:
            rgb = np.stack([filt.mean(0)] * 3, axis=-1)
        if rescale_per_filter:
            fmin, fmax = rgb.min(), rgb.max()
            if fmax > fmin:
                rgb = (rgb - fmin) / (fmax - fmin)
        else:
            rgb = np.clip(rgb, 0.0, 1.0)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"#{idx}", fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_per_channel(weights: torch.Tensor, path: Path, n_examples: int = 8) -> None:
    """Show the R/G/B planes of a few example filters separately."""
    w = weights.detach().cpu().numpy()
    nf, C, kH, kW = w.shape
    if C != 3:
        return
    rng = np.random.default_rng(0)
    idxs = rng.choice(nf, size=n_examples, replace=False)
    fig, axes = plt.subplots(n_examples, 4, figsize=(4 * 1.4, n_examples * 1.4))
    for row, idx in enumerate(idxs):
        filt = w[idx]
        rgb = filt.transpose(1, 2, 0)
        axes[row, 0].imshow(np.clip(rgb, 0, 1), interpolation="nearest")
        axes[row, 0].set_title(f"#{idx} RGB" if row == 0 else f"#{idx}", fontsize=7)
        for ci, label in enumerate(("R", "G", "B")):
            axes[row, 1 + ci].imshow(
                filt[ci], interpolation="nearest", cmap="gray", vmin=0.0, vmax=1.0
            )
            if row == 0:
                axes[row, 1 + ci].set_title(label, fontsize=7)
        for col in range(4):
            axes[row, col].axis("off")
    fig.suptitle("Random filters split into per-channel planes (raw [0,1])")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--top-k", type=int, default=32)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    weights = _load_encoder_weights(run_dir)
    print(f"loaded encoder weights: {tuple(weights.shape)}")

    out_dir = run_dir / "filter_analysis"
    out_dir.mkdir(exist_ok=True)

    l2, order, near0, near1 = _save_l2_summary(weights, out_dir / "l2_summary.png")
    print(f"L2 stats: min={l2.min():.3f} median={np.median(l2):.3f} max={l2.max():.3f}")
    print(
        f"per-filter mean fraction near 0: {near0.mean():.3f}; near 1: {near1.mean():.3f}"
    )

    _save_filter_grid(
        weights,
        order[: args.top_k],
        out_dir / "top_k_by_l2.png",
        title=f"Top-{args.top_k} filters by L2 (largest L2 first)",
    )
    _save_filter_grid(
        weights,
        order[-args.top_k :][::-1],
        out_dir / "bottom_k_by_l2.png",
        title=f"Bottom-{args.top_k} filters by L2 (smallest L2 first)",
    )
    _save_per_channel(weights, out_dir / "per_channel_examples.png")
    print(f"wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
