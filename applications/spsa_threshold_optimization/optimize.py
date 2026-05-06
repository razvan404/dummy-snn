"""SPSA threshold optimization for pre-trained SNNs.

Simultaneous Perturbation Stochastic Approximation (Spall 1992, 1998):
perturbs ALL thresholds simultaneously in a random Rademacher direction,
evaluates hard accuracy for both perturbations, estimates a gradient,
and takes a step. Only 2 feature extractions per iteration regardless
of the number of parameters.

Unlike gradient-based approaches (soft spike times, STE), SPSA directly
optimizes the true hard-feature accuracy — no differentiable approximation
needed. This sidesteps the discrete spike time landscape that defeats
sigmoid-based gradients.
"""

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from applications.common import load_split_data, resolve_model_dir, set_seed
from spiking.evaluation import evaluate_classifier
from spiking.evaluation.conv_feature_extraction import sum_pool_features
from spiking.evaluation.feature_extraction import spike_times_to_features
from spiking.utils.checkpoints import load_model

logger = logging.getLogger(__name__)


@dataclass
class SPSAConfig:
    """SPSA hyperparameters."""

    dataset: str = "cifar10"
    seed: int = 1
    num_filters: int = 256
    t_obj: float = 0.7

    # SPSA gain sequences: a_k = a / (k + A)^alpha, c_k = c / (k + 1)^gamma
    a: float = 0.5  # initial step size
    c: float = 0.10  # initial perturbation magnitude
    A: float = 1.0  # stabilization constant (typically 10% of max iters)
    alpha: float = 0.602  # step size decay exponent (Spall 1998)
    gamma: float = 0.101  # perturbation decay exponent (Spall 1998)

    steps: int = 10
    pool_size: int = 2
    device: str = "cuda"
    output_dir: str = ""


def extract_hard_features(
    layer,
    images: torch.Tensor,
    t_target: float,
    pool_size: int,
    chunk_size: int,
    device: str,
) -> np.ndarray:
    """Extract pooled features using hard spike times. Runs on GPU."""
    layer.eval()
    layer.to(device)
    N = len(images)
    flat_dim = layer.num_filters * pool_size * pool_size
    X = np.empty((N, flat_dim), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            st = layer.infer_spike_times_batch(images[start:end].to(device))
            feat = spike_times_to_features(st.cpu(), t_target=t_target)
            pooled = sum_pool_features(feat, pool_size)
            X[start:end] = pooled.flatten(1).numpy()

    layer.cpu()
    return X


def evaluate_thresholds(
    layer,
    thresholds: torch.Tensor,
    train_images: torch.Tensor,
    test_images: torch.Tensor,
    y_train: np.ndarray,
    y_test: np.ndarray,
    t_target: float,
    pool_size: int,
    device: str,
) -> tuple[float, float]:
    """Set thresholds, extract hard features, fit fresh SVC, return (train_acc, val_acc)."""
    layer.thresholds.data = thresholds.clone()
    X_train = extract_hard_features(
        layer, train_images, t_target, pool_size, chunk_size=2048, device=device
    )
    X_test = extract_hard_features(
        layer, test_images, t_target, pool_size, chunk_size=2048, device=device
    )
    train_metrics, val_metrics = evaluate_classifier(X_train, y_train, X_test, y_test)
    return train_metrics["accuracy"], val_metrics["accuracy"]


def multi_threshold_features(
    weights_4d: torch.Tensor,
    thresholds_list: list[torch.Tensor],
    images: torch.Tensor,
    t_target: float,
    pool_size: int,
    stride: int,
    padding: int,
    device: str,
    chunk_size: int = 256,
) -> list[np.ndarray]:
    """Extract features for multiple threshold vectors in ONE accumulation pass.

    The conv2d accumulation (expensive) runs once per chunk. Threshold crossing
    checks (cheap) run for each threshold vector simultaneously.
    Returns a list of (N, flat_dim) feature matrices, one per threshold vector.
    """
    N = len(images)
    num_thresholds = len(thresholds_list)
    num_filters = weights_4d.shape[0]
    kH = weights_4d.shape[2]
    oH = (images.shape[2] + 2 * padding - kH) // stride + 1
    oW = (images.shape[3] + 2 * padding - kH) // stride + 1
    flat_dim = num_filters * pool_size * pool_size

    # Stack thresholds: (num_thresholds, F)
    thresholds_2d = torch.stack(thresholds_list).to(device)
    thresholds_5d = thresholds_2d.view(num_thresholds, 1, num_filters, 1, 1)
    w = weights_4d.to(device)

    X_all = [np.empty((N, flat_dim), dtype=np.float32) for _ in range(num_thresholds)]

    with torch.no_grad():
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            inp = images[start:end].to(device)
            B = end - start

            # Init per-chunk
            result = torch.full(
                (num_thresholds, B, num_filters, oH, oW),
                float("inf"),
                dtype=inp.dtype,
                device=device,
            )
            not_yet_spiked = torch.ones(
                (num_thresholds, B, num_filters, oH, oW),
                dtype=torch.bool,
                device=device,
            )
            cum_potential = torch.zeros(
                (B, num_filters, oH, oW),
                dtype=inp.dtype,
                device=device,
            )

            finite_mask = torch.isfinite(inp)
            if finite_mask.any():
                unique_times = inp[finite_mask].unique().sort()[0]

                for t in unique_times:
                    active = (inp == t).float()
                    contrib = F.conv2d(active, w, stride=stride, padding=padding)
                    cum_potential += contrib

                    # Check ALL threshold vectors at once (cheap broadcast)
                    crossed = (
                        cum_potential.unsqueeze(0) >= thresholds_5d
                    ) & not_yet_spiked
                    result[crossed] = t
                    not_yet_spiked &= ~crossed

                    if not not_yet_spiked.any():
                        break

            # Convert spike times → features for each threshold vector
            for i in range(num_thresholds):
                feat = spike_times_to_features(result[i].cpu(), t_target=t_target)
                pooled = sum_pool_features(feat, pool_size)
                X_all[i][start:end] = pooled.flatten(1).numpy()

    return X_all


def evaluate_multi_thresholds(
    layer,
    thresholds_list: list[torch.Tensor],
    train_images: torch.Tensor,
    test_images: torch.Tensor,
    y_train: np.ndarray,
    y_test: np.ndarray,
    t_target: float,
    pool_size: int,
    device: str,
    chunk_size: int = 256,
) -> list[tuple[float, float]]:
    """Evaluate multiple threshold vectors in ONE forward pass. Returns [(train_acc, val_acc), ...]."""
    X_trains = multi_threshold_features(
        layer.weights_4d.detach(),
        thresholds_list,
        train_images,
        t_target,
        pool_size,
        layer.stride,
        layer.padding,
        device,
        chunk_size,
    )
    X_tests = multi_threshold_features(
        layer.weights_4d.detach(),
        thresholds_list,
        test_images,
        t_target,
        pool_size,
        layer.stride,
        layer.padding,
        device,
        chunk_size,
    )
    results = []
    for X_tr, X_te in zip(X_trains, X_tests):
        tr_m, val_m = evaluate_classifier(X_tr, y_train, X_te, y_test)
        results.append((tr_m["accuracy"], val_m["accuracy"]))
    return results


def plot_convergence(history: dict, baseline_acc: float, output_path: str) -> None:
    """Plot SPSA convergence: accuracy, gain sequences, threshold drift."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    steps = range(len(history["acc_after"]))

    # 1. Accuracy trajectory
    ax = axes[0, 0]
    ax.axhline(
        baseline_acc,
        color="green",
        linestyle="--",
        label=f"Baseline: {baseline_acc:.4f}",
    )
    ax.plot(steps, history["acc_plus"], "b^-", alpha=0.4, markersize=4, label="θ + cΔ")
    ax.plot(steps, history["acc_minus"], "rv-", alpha=0.4, markersize=4, label="θ - cΔ")
    ax.plot(
        steps,
        history["acc_after"],
        "ko-",
        linewidth=2,
        markersize=6,
        label="val acc",
    )
    if "train_after" in history and history["train_after"]:
        ax.plot(
            steps[: len(history["train_after"])],
            history["train_after"],
            "gs-",
            linewidth=1.5,
            markersize=5,
            alpha=0.7,
            label="train acc",
        )
    best_step = np.argmax(history["acc_after"])
    ax.axhline(
        history["acc_after"][best_step],
        color="orange",
        linestyle=":",
        label=f"Best: {history['acc_after'][best_step]:.4f} (step {best_step})",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("SVC Val Accuracy")
    ax.set_title("SPSA Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Gain sequences
    ax = axes[0, 1]
    ax.plot(steps, history["a_k"], "b-o", label="a_k (step size)", markersize=4)
    ax.plot(steps, history["c_k"], "r-o", label="c_k (perturbation)", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gain value")
    ax.set_title("Gain Sequences (Spall 1998)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Threshold drift
    ax = axes[1, 0]
    ax.plot(steps, history["drift"], "k-o", markersize=5)
    ax.set_xlabel("Step")
    ax.set_ylabel("L2 drift from original")
    ax.set_title("Threshold Drift")
    ax.grid(True, alpha=0.3)

    # 4. Accuracy delta per step
    ax = axes[1, 1]
    acc = history["acc_after"]
    deltas = [acc[0] - baseline_acc] + [acc[i] - acc[i - 1] for i in range(1, len(acc))]
    colors = ["green" if d > 0 else "red" for d in deltas]
    ax.bar(steps, deltas, color=colors, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Δ Accuracy")
    ax.set_title("Per-Step Accuracy Change")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info("Plot saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="SPSA threshold optimization")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-filters", type=int, default=256)
    parser.add_argument("--t-obj", type=float, default=0.7)
    parser.add_argument("--a", type=float, default=0.5, help="Initial step size")
    parser.add_argument("--c", type=float, default=0.10, help="Initial perturbation")
    parser.add_argument("--A", type=float, default=1.0, help="Stabilization constant")
    parser.add_argument("--alpha", type=float, default=0.602)
    parser.add_argument("--gamma", type=float, default=0.101)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--resume",
        default="",
        help="Path to previous results.json to resume from (loads best thresholds + history)",
    )
    args = parser.parse_args()

    config = SPSAConfig(
        **{k: v for k, v in vars(args).items() if k in SPSAConfig.__dataclass_fields__}
    )
    set_seed(config.seed)
    device = config.device

    # Resolve paths
    model_dir = resolve_model_dir(
        config.dataset, config.num_filters, config.t_obj, config.seed
    )
    model_path = f"{model_dir}/model.pth"
    if not os.path.exists(model_path):
        logger.error("No model at %s", model_path)
        return

    with open(f"{model_dir}/setup.json") as f:
        t_target = json.load(f).get("target_timestamp", config.t_obj)

    output_dir = config.output_dir or f"{model_dir}/spsa_opt"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(asdict(config), f, indent=4)

    # Load model and data
    logger.info("Loading model from %s", model_path)
    layer = load_model(model_path)

    logger.info("Loading data...")
    train_data, test_data = load_split_data(config.dataset)
    train_images, train_labels = train_data["images"], train_data["labels"]
    test_images, test_labels = test_data["images"], test_data["labels"]
    y_train = train_labels.numpy()
    y_test = test_labels.numpy()
    logger.info("Train: %d, Test: %d", len(train_images), len(test_images))

    # Baseline
    original_thresholds = layer.thresholds.detach().clone()
    logger.info("Evaluating baseline...")
    baseline_train, baseline_acc = evaluate_thresholds(
        layer,
        original_thresholds,
        train_images,
        test_images,
        y_train,
        y_test,
        t_target,
        config.pool_size,
        device,
    )
    logger.info("Baseline SVC — train: %.4f, val: %.4f", baseline_train, baseline_acc)

    # SPSA optimization — resume or start fresh
    resume_path = args.resume
    start_step = 0
    if resume_path and os.path.exists(resume_path):
        with open(resume_path) as f:
            prev = json.load(f)
        theta = torch.tensor(prev["best_thresholds"], dtype=torch.float32)
        best_theta = theta.clone()
        best_acc = prev["best_acc"]
        start_step = len(prev["history"]["acc_after"])
        history = prev["history"]
        logger.info(
            "Resumed from %s — step %d, best acc: %.4f",
            resume_path,
            start_step,
            best_acc,
        )
    else:
        theta = original_thresholds.clone()
        best_theta = theta.clone()
        best_acc = baseline_acc
        history = {
            "acc_plus": [],
            "acc_minus": [],
            "acc_after": [],
            "train_after": [],
            "a_k": [],
            "c_k": [],
            "drift": [],
        }

    num_params = len(theta)
    total_steps = start_step + config.steps

    logger.info(
        "Starting SPSA: steps %d→%d, a=%.3f, c=%.3f",
        start_step + 1,
        total_steps,
        config.a,
        config.c,
    )

    for step_idx in range(config.steps):
        k = step_idx  # gain sequences always start from 0 (fresh schedule)
        global_step = start_step + step_idx  # for logging/history only

        # Gain sequences (Spall 1998) — reset on resume
        a_k = config.a / (k + 1 + config.A) ** config.alpha
        c_k = config.c / (k + 1) ** config.gamma

        # Rademacher perturbation: each component ±1 independently
        delta = 2 * torch.randint(0, 2, (num_params,), dtype=torch.float32) - 1

        # Evaluate perturbed + updated points in ONE accumulation pass
        theta_plus = theta + c_k * delta
        theta_minus = theta - c_k * delta

        # Gradient estimation: evaluate θ+cΔ and θ-cΔ together (1 pass)
        (_, acc_plus), (_, acc_minus) = evaluate_multi_thresholds(
            layer,
            [theta_plus, theta_minus],
            train_images,
            test_images,
            y_train,
            y_test,
            t_target,
            config.pool_size,
            device,
        )

        # SPSA gradient estimate (maximize accuracy → ascent)
        g_k = (acc_plus - acc_minus) / (2 * c_k * delta)
        theta = theta + a_k * g_k

        # Evaluate after update (separate pass — need fresh SVC)
        train_after, acc_after = evaluate_thresholds(
            layer,
            theta,
            train_images,
            test_images,
            y_train,
            y_test,
            t_target,
            config.pool_size,
            device,
        )

        # Track best
        if acc_after > best_acc:
            best_acc = acc_after
            best_theta = theta.clone()

        drift = (theta - original_thresholds).norm().item()

        history["acc_plus"].append(acc_plus)
        history["acc_minus"].append(acc_minus)
        history["acc_after"].append(acc_after)
        history["train_after"].append(train_after)
        history["a_k"].append(a_k)
        history["c_k"].append(c_k)
        history["drift"].append(drift)

        logger.info(
            "Step %2d/%d | a_k=%.4f c_k=%.4f | acc+: %.4f | acc-: %.4f | "
            "train: %.4f | val: %.4f (%+.4f) | drift: %.3f",
            global_step + 1,
            total_steps,
            a_k,
            c_k,
            acc_plus,
            acc_minus,
            train_after,
            acc_after,
            acc_after - baseline_acc,
            drift,
        )

    # Final evaluation with best thresholds
    logger.info("=== Final Results ===")
    logger.info(
        "Best SVC val: %.4f (%+.4f vs baseline)", best_acc, best_acc - baseline_acc
    )

    # Save results
    results = {
        "baseline_acc": baseline_acc,
        "best_acc": best_acc,
        "best_step": int(np.argmax(history["acc_after"])),
        "original_thresholds": original_thresholds.tolist(),
        "best_thresholds": best_theta.tolist(),
        "final_thresholds": theta.tolist(),
        "history": history,
        "config": asdict(config),
    }
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Results saved to %s/results.json", output_dir)

    # Plot
    plot_convergence(history, baseline_acc, f"{output_dir}/convergence.png")

    logger.info("=== Summary ===")
    logger.info("Baseline: %.4f", baseline_acc)
    logger.info(
        "Best:     %.4f (%+.4f, step %d)",
        best_acc,
        best_acc - baseline_acc,
        results["best_step"],
    )
    logger.info(
        "Final:    %.4f (%+.4f)",
        history["acc_after"][-1],
        history["acc_after"][-1] - baseline_acc,
    )
