import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from applications.common import set_seed
from applications.paper_hyperparams import get_paper_hyperparams
from spiking import (
    BiologicalSTDP,
    ConvIntegrateAndFireLayer,
    ConvLearner,
    ConvUnsupervisedTrainer,
    MultiplicativeSTDP,
    NormalInitialization,
    SequentialThresholdAdaptation,
    WinnerTakesAll,
    CompetitiveThresholdAdaptation,
    TargetTimestampAdaptation,
    save_model,
)

logger = logging.getLogger(__name__)


def _create_stdp(variant: str, params: dict) -> MultiplicativeSTDP | BiologicalSTDP:
    """Create STDP learning mechanism from paper hyperparams."""
    lr = params["stdp_lr"]
    annealing = params["annealing"]
    w_min = params["w_min"]
    w_max = params["w_max"]
    if variant == "multiplicative":
        return MultiplicativeSTDP(
            learning_rate=lr,
            decay_factor=annealing,
            beta=params.get("beta", 1.0),
            w_min=w_min,
            w_max=w_max,
        )
    elif variant == "biological":
        return BiologicalSTDP(
            tau_pre=params.get("biological_tau", 0.1),
            tau_post=params.get("biological_tau", 0.1),
            max_pre_spike_time=1.0,
            learning_rate=lr,
            decay_factor=annealing,
            weights_interval=(w_min, w_max),
        )
    raise ValueError(f"Unknown STDP variant: {variant!r}")


def _create_threshold_adaptation(params: dict) -> SequentialThresholdAdaptation:
    """Create threshold adaptation (competitive + target timestamp)."""
    return SequentialThresholdAdaptation(
        [
            CompetitiveThresholdAdaptation(
                min_threshold=params["min_threshold"],
                learning_rate=params["threshold_lr"],
                decay_factor=params["annealing"],
            ),
            TargetTimestampAdaptation(
                target_timestamp=params["target_timestamp"],
                min_threshold=params["min_threshold"],
                learning_rate=params["threshold_lr"],
                decay_factor=params["annealing"],
            ),
        ]
    )


def _save_filter_grid(weights_4d: torch.Tensor, path: str, ncols: int = 16):
    """Save learned filter weights as a grid image."""
    num_filters, C, kH, kW = weights_4d.shape
    nrows = (num_filters + ncols - 1) // ncols
    w = weights_4d.detach().cpu().numpy()

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        ax.axis("off")
        if i >= num_filters:
            continue
        filt = w[i]  # (C, kH, kW)
        if C == 2:
            on, off = filt[0], filt[1]
            rgb = np.stack([off, on, np.zeros_like(on)], axis=-1)
        elif C >= 6:
            rgb = filt[[0, 2, 4]].transpose(1, 2, 0)
        else:
            rgb = np.stack([filt.mean(0)] * 3, axis=-1)
        fmin, fmax = rgb.min(), rgb.max()
        if fmax > fmin:
            rgb = (rgb - fmin) / (fmax - fmin)
        ax.imshow(rgb, interpolation="nearest")

    plt.tight_layout(pad=0.1)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_training_summary(
    layer, neuron_wins: torch.Tensor, output_dir: str, num_filters: int
):
    """Save end-of-training summary: neuron win distribution + weight grid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Panel 1: Neuron win distribution
    ax = axes[0]
    wins = neuron_wins.numpy()
    colors = ["C3" if w == 0 else "C0" for w in wins]
    ax.bar(range(num_filters), wins, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("Win count")
    dead = int((wins == 0).sum())
    ax.set_title(f"Neuron win distribution ({dead} dead / {num_filters} total)")

    # Panel 2: Weight norms per filter
    ax = axes[1]
    w4d = layer.weights_4d.detach().cpu()  # (nf, C, kH, kW)
    norms = w4d.flatten(1).norm(dim=1).numpy()
    ax.bar(range(num_filters), norms, color="C2", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("L2 norm")
    ax.set_title("Filter weight norms")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _extract_random_patches(images: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Extract one random patch per image as spatial spike times.

    :param images: (N, C, H, W) encoded spike times.
    :param kernel_size: Patch side length.
    :returns: (N, C, kH, kW) spatial patches.
    """
    N, C, H, W = images.shape
    max_row = H - kernel_size
    max_col = W - kernel_size
    rows = torch.randint(0, max_row + 1, (N,))
    cols = torch.randint(0, max_col + 1, (N,))

    patches = torch.empty(N, C, kernel_size, kernel_size)
    for i in range(N):
        patches[i] = images[
            i, :, rows[i] : rows[i] + kernel_size, cols[i] : cols[i] + kernel_size
        ]
    return patches


def _load_training_images(dataset: str, processed_dir: str | None) -> torch.Tensor:
    """Load training spike times, using dataset classes when possible.

    For cifar10 without a processed_dir, uses Cifar10WhitenedDataset (rho=1.0).
    Otherwise loads from the preprocessed .pt files.
    """
    if dataset == "cifar10" and processed_dir is None:
        from applications.datasets import Cifar10WhitenedDataset

        ds = Cifar10WhitenedDataset("data", "train")
        return ds.all_times
    if processed_dir is None:
        processed_dir = f"data/processed-{dataset}"
    train_data = torch.load(f"{processed_dir}/train.pt", weights_only=True)
    return train_data["images"]


def train_model(
    *,
    dataset: str,
    seed: int,
    t_obj: float | None = None,
    num_filters: int | None = None,
    num_epochs: int | None = None,
    processed_dir: str | None = None,
    output_dir: str,
    params_override: dict | None = None,
) -> dict:
    """Train a conv SNN on preprocessed data (training only, no evaluation).

    Loads preprocessed spike-encoded images, trains on random patches per epoch
    (Falez 2019/2020 methodology), saves model, setup, and training logs.

    :param dataset: 'mnist' or 'cifar10'.
    :param seed: Random seed.
    :param t_obj: Target timestamp override (sweepable).
    :param num_filters: Number of convolutional filters (overrides paper default).
    :param num_epochs: Number of epochs (overrides paper default).
    :param processed_dir: Path to preprocessed data directory.
    :param output_dir: Where to save model, setup, and weights visualization.
    :param params_override: Additional hyperparameter overrides.
    :returns: Dict with setup info and training logs.
    """
    params = get_paper_hyperparams(dataset)
    if params_override:
        params.update(params_override)
    if t_obj is not None:
        params["target_timestamp"] = t_obj
    if num_filters is not None:
        params["num_filters"] = num_filters
    if num_epochs is not None:
        params["num_epochs"] = num_epochs

    set_seed(seed)
    nf = params["num_filters"]
    ne = params["num_epochs"]
    tt = params["target_timestamp"]
    stdp_variant = params["stdp_variant"]

    logger.info(
        "Training: %s, %s STDP, %d filters, t_obj=%.2f, %d epochs, seed=%d",
        dataset,
        stdp_variant,
        nf,
        tt,
        ne,
        seed,
    )

    # Load data
    logger.info("Loading training data for %s...", dataset)
    all_images = _load_training_images(dataset, processed_dir)
    N = len(all_images)
    in_channels = all_images.shape[1]
    ksize = params["kernel_size"]
    logger.info("  %d images, %d channels, kernel=%d", N, in_channels, ksize)

    # Create model
    init = NormalInitialization(
        avg_threshold=params["threshold_avg"],
        min_threshold=params["min_threshold"],
        std_dev=params["threshold_std"],
    )
    layer = ConvIntegrateAndFireLayer(
        in_channels=in_channels,
        num_filters=nf,
        kernel_size=ksize,
        stride=params["stride"],
        padding=params["padding"],
        threshold_initialization=init,
        refractory_period=float("inf"),
    )
    torch.nn.init.uniform_(layer.weights, a=params["w_min"], b=params["w_max"])

    # Create learner
    stdp = _create_stdp(stdp_variant, params)
    adaptation = _create_threshold_adaptation(params)
    learner = ConvLearner(
        layer, stdp, competition=WinnerTakesAll(), threshold_adaptation=adaptation
    )
    trainer = ConvUnsupervisedTrainer(
        layer, learner, image_shape=(in_channels, ksize, ksize), early_stopping=True
    )

    # Training with per-epoch log collection
    training_logs = {
        "epoch_mean_dw": [],
        "epoch_threshold_mean": [],
        "epoch_threshold_std": [],
        "epoch_threshold_min": [],
        "epoch_threshold_max": [],
        "epoch_thresholds": [],  # full threshold vector per epoch
    }

    # Track neuron win counts and last-10k spike activity
    neuron_wins = torch.zeros(nf, dtype=torch.long)
    total_steps = N * ne
    log_last_n = 10_000
    last10k_winners = torch.full((log_last_n,), -1, dtype=torch.long)
    last10k_spike_times = torch.full((log_last_n,), float("inf"))
    global_step = 0

    for epoch in tqdm(range(ne), desc="Training", unit="epoch"):
        patches = _extract_random_patches(all_images, ksize)
        perm = torch.randperm(N)
        layer.train()
        epoch_dws = []
        it = tqdm(range(N), desc=f"epoch {epoch}", unit="patch", leave=False)
        for i in it:
            dw = trainer.step_batch(i, patches[perm[i]])
            epoch_dws.append(dw)

            if learner.neurons_to_learn is not None:
                for idx in learner.neurons_to_learn:
                    neuron_wins[idx.item()] += 1

            # Record winner and spike time for the last log_last_n steps
            steps_remaining = total_steps - global_step
            if steps_remaining <= log_last_n:
                slot = log_last_n - steps_remaining
                if (
                    learner.neurons_to_learn is not None
                    and len(learner.neurons_to_learn) > 0
                ):
                    winner = learner.neurons_to_learn[0].item()
                else:
                    winner = -1
                last10k_winners[slot] = winner
                last10k_spike_times[slot] = learner.winner_spike_time

            global_step += 1

        trainer.step_epoch()

        # Record epoch stats
        thresholds = layer.thresholds.detach().cpu()
        training_logs["epoch_mean_dw"].append(float(np.mean(epoch_dws)))
        training_logs["epoch_threshold_mean"].append(float(thresholds.mean()))
        training_logs["epoch_threshold_std"].append(float(thresholds.std()))
        training_logs["epoch_threshold_min"].append(float(thresholds.min()))
        training_logs["epoch_threshold_max"].append(float(thresholds.max()))
        training_logs["epoch_thresholds"].append(thresholds.clone())

    # Convert threshold list to tensor for compact storage
    training_logs["epoch_thresholds"] = torch.stack(
        training_logs["epoch_thresholds"]
    )  # (num_epochs, num_filters)
    training_logs["neuron_wins"] = neuron_wins  # (num_filters,)
    training_logs["last10k_winners"] = (
        last10k_winners  # (10000,) winner neuron index, -1 = no spike
    )
    training_logs["last10k_spike_times"] = (
        last10k_spike_times  # (10000,) spike time of winner
    )

    # Save model, setup, and logs
    os.makedirs(output_dir, exist_ok=True)
    save_model(layer, f"{output_dir}/model.pth")
    torch.save(training_logs, f"{output_dir}/training_logs.pt")

    # End-of-training plots: neuron win distribution + learned weights
    _save_training_summary(layer, neuron_wins, output_dir, nf)
    _save_filter_grid(layer.weights_4d, f"{output_dir}/weights.png", ncols=min(16, nf))

    setup_info = {
        "dataset": dataset,
        "seed": seed,
        "processed_dir": processed_dir,
        **params,
    }
    with open(f"{output_dir}/setup.json", "w") as f:
        json.dump(setup_info, f, indent=4)

    logger.info("Saved model and logs to %s", output_dir)
    return {**setup_info, "training_logs": training_logs}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Train conv SNN on preprocessed dataset (training only)"
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["mnist", "cifar10"],
        help="Dataset name",
    )
    parser.add_argument("--num-filters", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--t-obj", type=float, default=None)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1],
        help="Random seeds to train (e.g. --seeds 1 2 3 4 5 6 7 8 9 10)",
    )
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if model already exists",
    )
    args = parser.parse_args()

    hp = get_paper_hyperparams(args.dataset)
    nf = args.num_filters or hp["num_filters"]
    t = args.t_obj or hp["target_timestamp"]

    if args.base_dir is None:
        if args.dataset == "cifar10":
            args.base_dir = "logs/cifar10_whitened/sweep"
        else:
            args.base_dir = f"logs/{args.dataset}/sweep"

    for seed in args.seeds:
        output_dir = f"{args.base_dir}/nf_{nf}/tobj_{t:.2f}/seed_{seed}"
        if not args.force and os.path.exists(f"{output_dir}/model.pth"):
            logger.info(
                "Skipping seed %d (already trained, use --force to retrain)", seed
            )
            continue
        train_model(
            dataset=args.dataset,
            seed=seed,
            t_obj=args.t_obj,
            num_filters=args.num_filters,
            num_epochs=args.num_epochs,
            processed_dir=args.processed_dir,
            output_dir=output_dir,
        )
