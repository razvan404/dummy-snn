"""Visualizations for the ANN bimodal autoencoder.

All plots are saved as PNG; nothing is shown interactively.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_weight_histogram(
    weights: torch.Tensor,
    path: str | Path,
    title: str,
    bins: int = 80,
    xlim: tuple[float, float] = (-0.05, 1.05),
) -> None:
    """Histogram of all encoder weights with markers at 0 and 1."""
    w = weights.detach().cpu().numpy().ravel()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(w, bins=bins, range=xlim, color="C0", edgecolor="black", linewidth=0.3)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlim(*xlim)
    ax.set_xlabel("encoder weight")
    ax.set_ylabel("count")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_filter_grid(
    weights_4d: torch.Tensor,
    path: str | Path,
    ncols: int = 16,
    rescale_per_filter: bool = True,
) -> None:
    """Render encoder filters as an RGB grid.

    :param weights_4d: (num_filters, C, kH, kW) tensor.
    :param rescale_per_filter: if True, min-max normalize each filter for
        contrast; if False, plot raw weights (exposes the [0,1] range).
    """
    nf, C, kH, kW = weights_4d.shape
    nrows = (nf + ncols - 1) // ncols
    w = weights_4d.detach().cpu().numpy()

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        ax.axis("off")
        if i >= nf:
            continue
        filt = w[i]  # (C, kH, kW)
        if C == 3:
            rgb = filt.transpose(1, 2, 0)
        elif C == 1:
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

    plt.tight_layout(pad=0.1)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_filter_grid_whitened(
    weights_4d: torch.Tensor,
    path: str | Path,
    ncols: int = 16,
) -> None:
    """Render 6-channel (whitened ON/OFF) encoder filters as RGB.

    Channel layout follows ``encode_whitened_image``: ``[R+, R-, G+, G-, B+, B-]``.
    For each filter we form ``R = w[0] - w[1]``, ``G = w[2] - w[3]``,
    ``B = w[4] - w[5]`` (range [-1, 1] given encoder clipped to [0, 1]) and
    rescale per-filter to [0, 1] so positive-channel dominance reads as that
    color and negative-channel dominance reads as the complement.
    """
    nf, C, kH, kW = weights_4d.shape
    if C != 6:
        raise ValueError(f"expected 6 input channels, got {C}")
    nrows = (nf + ncols - 1) // ncols
    w = weights_4d.detach().cpu().numpy()
    diffs = np.stack([w[:, 0] - w[:, 1], w[:, 2] - w[:, 3], w[:, 4] - w[:, 5]], axis=1)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        ax.axis("off")
        if i >= nf:
            continue
        rgb = diffs[i].transpose(1, 2, 0)  # (H, W, 3) in [-1, 1]
        fmin, fmax = rgb.min(), rgb.max()
        if fmax > fmin:
            rgb = (rgb - fmin) / (fmax - fmin)
        else:
            rgb = np.full_like(rgb, 0.5)
        ax.imshow(rgb, interpolation="nearest")

    plt.tight_layout(pad=0.1)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_reconstructions(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    path: str | Path,
    n: int = 8,
) -> None:
    """Side-by-side: top row originals, bottom row reconstructions."""
    n = min(n, originals.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(n * 1.4, 3.0))
    for i in range(n):
        for row, img in enumerate((originals[i], reconstructions[i])):
            arr = img.detach().cpu().numpy()
            if arr.shape[0] == 1:
                arr = np.stack([arr[0]] * 3, axis=-1)
            else:
                arr = arr.transpose(1, 2, 0)
            arr = np.clip(arr, 0.0, 1.0)
            axes[row, i].imshow(arr, interpolation="nearest")
            axes[row, i].axis("off")
    axes[0, 0].set_title("target", loc="left", fontsize=9)
    axes[1, 0].set_title("recon", loc="left", fontsize=9)
    plt.tight_layout(pad=0.2)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_loss_curves(history: dict, path: str | Path) -> None:
    """Plot recon, bimodal, and total loss across epochs."""
    epochs = np.arange(1, len(history["recon_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.plot(epochs, history["recon_loss"], color="C0", label="RMSE (recon)")
    ax1.plot(epochs, history["total_loss"], color="C1", label="total")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["bimodal_loss"], color="C3", label="bimodal pen.")
    ax2.set_ylabel("bimodal penalty", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    plt.title("Training losses")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_bimodality_curve(history: dict, path: str | Path) -> None:
    """Fraction of weights near {0, 1} per epoch."""
    epochs = np.arange(1, len(history["bimodality_score"]) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["bimodality_score"], color="C2", marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("fraction within 0.1 of {0, 1}")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Encoder weight bimodality over training")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_lambda_sweep(sweep_records: list[dict], path: str | Path) -> None:
    """RMSE vs final bimodality score, annotated with λ values."""
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [r["final_bimodality"] for r in sweep_records]
    ys = [r["final_recon_rmse"] for r in sweep_records]
    labels = [f"λ={r['lambda_bimodal']}" for r in sweep_records]
    ax.scatter(xs, ys, color="C0", s=60, zorder=3)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("bimodality score (fraction within 0.1 of {0,1})")
    ax.set_ylabel("final reconstruction RMSE")
    ax.set_title("Bimodality vs reconstruction trade-off")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
