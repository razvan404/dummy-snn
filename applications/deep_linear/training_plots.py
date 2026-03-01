import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch


def save_winner_counts(win_counts: torch.Tensor, path: str):
    """Save a bar chart of per-neuron WTA win frequency."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(win_counts)), win_counts.numpy())
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Win count")
    ax.set_title("Winner counts")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_threshold_distribution(thresholds: torch.Tensor, path: str):
    """Save a histogram of threshold values."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(thresholds.detach().cpu().numpy(), bins=30)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Count")
    ax.set_title("Threshold distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
