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
from applications.pipeline.datasets import dataset_names, load_train_images
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
    WeightMeanAdaptation,
    save_model,
)

logger = logging.getLogger(__name__)


def _create_stdp(variant: str, params: dict) -> MultiplicativeSTDP | BiologicalSTDP:
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
    mode = params.get("threshold_mode", "falez")
    competitive = CompetitiveThresholdAdaptation(
        min_threshold=params["min_threshold"],
        learning_rate=params["threshold_lr"],
        decay_factor=params["annealing"],
    )
    if mode == "falez":
        return SequentialThresholdAdaptation(
            [
                competitive,
                TargetTimestampAdaptation(
                    target_timestamp=params["target_timestamp"],
                    min_threshold=params["min_threshold"],
                    learning_rate=params["threshold_lr"],
                    decay_factor=params["annealing"],
                ),
            ]
        )
    if mode == "weight_mean":
        return SequentialThresholdAdaptation(
            [
                competitive,
                WeightMeanAdaptation(
                    min_threshold=params["min_threshold"],
                    learning_rate=params["threshold_lr"],
                    target_mean=params.get("target_mean", 0.5),
                    decay_factor=params["annealing"],
                    max_threshold=params.get("max_threshold"),
                    use_winner=params.get("use_winner_mean", False),
                ),
            ]
        )
    raise ValueError(f"Unknown threshold_mode: {mode!r}")


def _save_filter_grid(weights_4d: torch.Tensor, path: str, ncols: int = 16):
    ...
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
        filt = w[i]
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
    ...
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    wins = neuron_wins.numpy()
    colors = ["C3" if w == 0 else "C0" for w in wins]
    ax.bar(range(num_filters), wins, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("Win count")
    dead = int((wins == 0).sum())
    ax.set_title(f"Neuron win distribution ({dead} dead / {num_filters} total)")

    ax = axes[1]
    w4d = layer.weights_4d.detach().cpu()
    norms = w4d.flatten(1).norm(dim=1).numpy()
    ax.bar(range(num_filters), norms, color="C2", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("L2 norm")
    ax.set_title("Filter weight norms")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _extract_random_patches(images: torch.Tensor, kernel_size: int) -> torch.Tensor:
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


def _load_training_images(
    dataset: str, processed_dir: str | None, num_bins: int = 64
) -> torch.Tensor:
    if processed_dir is None:
        return load_train_images(dataset, num_bins)
    train_data = torch.load(f"{processed_dir}/train.pt", weights_only=True)
    return train_data["images"]


def _build_and_train_layer(train_images: torch.Tensor, params: dict, *, device: str = "cpu"):
    """Train one conv-IF layer by STDP on random patches of `train_images`.

    The reusable training core: works on raw encoded images (layer 1) or on a frozen
    prefix's featurized spike-time maps (layer 2). Caller is responsible for seeding.
    `device` runs the per-sample STDP loop on GPU; backend (default gather) is read from
    params["backend"].
    """
    nf = params["num_filters"]
    ne = params["num_epochs"]
    in_channels = train_images.shape[1]
    ksize = params["kernel_size"]
    logger.info(
        "  layer: %d filters, %s STDP, %d ch, kernel=%d, t_obj=%.2f, %d epochs, %d images",
        nf, params["stdp_variant"], in_channels, ksize,
        params["target_timestamp"], ne, len(train_images),
    )

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
        backend=params.get("backend", "gather"),
    )
    torch.nn.init.uniform_(layer.weights, a=params["w_min"], b=params["w_max"])
    layer.num_bins = params["num_bins"]

    stdp = _create_stdp(params["stdp_variant"], params)
    adaptation = _create_threshold_adaptation(params)
    learner = ConvLearner(
        layer, stdp, competition=WinnerTakesAll(), threshold_adaptation=adaptation
    )
    trainer = ConvUnsupervisedTrainer(
        layer, learner, image_shape=(in_channels, ksize, ksize),
        early_stopping=True, device=device,
    )

    training_logs = {
        "epoch_mean_dw": [],
        "epoch_threshold_mean": [],
        "epoch_threshold_std": [],
        "epoch_threshold_min": [],
        "epoch_threshold_max": [],
        "epoch_thresholds": [],
    }

    dev = layer.weights.device
    N = len(train_images)
    neuron_wins = torch.zeros(nf, dtype=torch.long, device=dev)
    total_steps = N * ne
    log_last_n = 10_000
    last10k_winners = torch.full((log_last_n,), -1, dtype=torch.long, device=dev)
    last10k_spike_times = torch.full((log_last_n,), float("inf"), device=dev)
    global_step = 0

    for epoch in tqdm(range(ne), desc="Training", unit="epoch"):
        patches = _extract_random_patches(train_images, ksize)
        perm = torch.randperm(N)
        layer.train()
        epoch_dws = torch.empty(N, device=dev)
        for i in range(N):
            dw = trainer.step_batch(i, patches[perm[i]])
            epoch_dws[i] = dw

            ntl = learner.neurons_to_learn
            if ntl is not None and ntl.numel() > 0:
                neuron_wins.scatter_add_(0, ntl, torch.ones_like(ntl, dtype=torch.long))

            steps_remaining = total_steps - global_step
            if steps_remaining <= log_last_n:
                slot = log_last_n - steps_remaining
                if ntl is not None and ntl.numel() > 0:
                    last10k_winners[slot] = ntl[0]
                last10k_spike_times[slot] = learner.winner_spike_time

            global_step += 1

        trainer.step_epoch()

        epoch_dws_cpu = epoch_dws.detach().cpu()
        thresholds = layer.thresholds.detach().cpu()
        training_logs["epoch_mean_dw"].append(float(epoch_dws_cpu.mean()))
        training_logs["epoch_threshold_mean"].append(float(thresholds.mean()))
        training_logs["epoch_threshold_std"].append(float(thresholds.std()))
        training_logs["epoch_threshold_min"].append(float(thresholds.min()))
        training_logs["epoch_threshold_max"].append(float(thresholds.max()))
        training_logs["epoch_thresholds"].append(thresholds.clone())

    training_logs["epoch_thresholds"] = torch.stack(training_logs["epoch_thresholds"])
    training_logs["neuron_wins"] = neuron_wins.cpu()
    training_logs["last10k_winners"] = last10k_winners.cpu()
    training_logs["last10k_spike_times"] = last10k_spike_times.cpu()
    return layer, training_logs


def _save_run(layer, training_logs, params, dataset, seed, processed_dir, output_dir):
    nf = params["num_filters"]
    os.makedirs(output_dir, exist_ok=True)
    save_model(layer, f"{output_dir}/model.pth")
    torch.save(training_logs, f"{output_dir}/training_logs.pt")

    _save_training_summary(layer, training_logs["neuron_wins"], output_dir, nf)
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
    return setup_info


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
    logger.info(
        "Training: %s, %s STDP, %d filters, t_obj=%.2f, %d epochs, seed=%d",
        dataset, params["stdp_variant"], params["num_filters"],
        params["target_timestamp"], params["num_epochs"], seed,
    )
    logger.info("Loading training data for %s...", dataset)
    all_images = _load_training_images(dataset, processed_dir, params["num_bins"])
    layer, training_logs = _build_and_train_layer(all_images, params)
    setup_info = _save_run(
        layer, training_logs, params, dataset, seed, processed_dir, output_dir
    )
    return {**setup_info, "training_logs": training_logs}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Train conv SNN on preprocessed dataset (training only)"
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=dataset_names(),
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
        "--threshold-mode",
        type=str,
        default="falez",
        choices=["falez", "weight_mean"],
        help="Threshold-adaptation rule (default: 'falez' = competitive + target-timestamp).",
    )
    parser.add_argument(
        "--target-mean",
        type=float,
        default=0.5,
        help="Target per-filter mean weight when --threshold-mode=weight_mean.",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=None,
        help="Upper clamp on thresholds (weight_mean mode). Default: no cap.",
    )
    parser.add_argument(
        "--use-winner-mean",
        action="store_true",
        help="weight_mean: drive ALL thresholds by the winner's mean weight (Falez Eq 6 style).",
    )
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
        dir_name = "cifar10_whitened" if args.dataset == "cifar10" else args.dataset
        suffix = "_wmean" if args.threshold_mode == "weight_mean" else ""
        args.base_dir = f"logs/{dir_name}/sweep{suffix}"

    overrides = {"threshold_mode": args.threshold_mode}
    if args.threshold_mode == "weight_mean":
        overrides["target_mean"] = args.target_mean
        if args.max_threshold is not None:
            overrides["max_threshold"] = args.max_threshold
        if args.use_winner_mean:
            overrides["use_winner_mean"] = True

    if args.threshold_mode == "weight_mean":
        param_seg = f"wtarget_{args.target_mean:.2f}"
    else:
        param_seg = f"tobj_{t:.2f}"

    for seed in args.seeds:
        output_dir = f"{args.base_dir}/nf_{nf}/{param_seg}/seed_{seed}"
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
            params_override=overrides,
        )
