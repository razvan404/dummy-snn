"""Response Surface Methodology (RSM) for SNN threshold optimization.

Instead of optimizing thresholds one at a time (sequential greedy) or with
noisy gradient estimates (SPSA), RSM:
1. Evaluates ~300 carefully designed threshold configurations in a few forward passes
2. Fits a polynomial model: acc ≈ β₀ + Σ βᵢδᵢ + Σ βᵢⱼδᵢδⱼ
3. Reads off optimal thresholds from the fitted surface

The multi-threshold evaluation trick (one conv2d pass, many threshold checks)
makes step 1 cheap. The fitted model captures neuron interactions (βᵢⱼ terms)
that sequential greedy misses.
"""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from applications.common import load_split_data, resolve_model_dir, set_seed
from spiking.evaluation.conv_feature_extraction import sum_pool_features
from spiking.evaluation.feature_extraction import spike_times_to_features
from spiking.utils.checkpoints import load_model

logger = logging.getLogger(__name__)


@dataclass
class RSMConfig:
    dataset: str = "cifar10"
    seed: int = 1
    num_filters: int = 256
    t_obj: float = 0.7

    # Design
    n_configs: int = 300  # number of random threshold configurations
    perturbation_scale: float = 0.05  # each θ perturbed by ±scale*θ

    # Model fitting
    fit_interactions: bool = False  # fit pairwise βᵢⱼ (needs more configs)
    ridge_alpha_fit: float = 1.0  # regularization for the RSM polynomial fit

    # Classifier for evaluation
    classifier: str = "ridge"  # "ridge" or "svc"
    classifier_alpha: float = 1.0  # Ridge alpha (if classifier=ridge)

    # Compute
    pool_size: int = 2
    config_batch_size: int = 50  # configs per multi-threshold pass
    chunk_size: int = 16  # images per chunk within multi-threshold
    device: str = "cuda"
    output_dir: str = ""


# ---------------------------------------------------------------------------
# Design matrix generation
# ---------------------------------------------------------------------------


def generate_rademacher_design(n_configs: int, n_params: int, seed: int) -> np.ndarray:
    """Generate a random ±1 design matrix (Rademacher).

    :returns: (n_configs, n_params) array of ±1.
    """
    rng = np.random.RandomState(seed)
    return rng.choice([-1, 1], size=(n_configs, n_params)).astype(np.float32)


# ---------------------------------------------------------------------------
# Multi-threshold feature extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def multi_threshold_features(
    weights_4d: torch.Tensor,
    thresholds_2d: torch.Tensor,
    images: torch.Tensor,
    t_target: float,
    pool_size: int,
    stride: int,
    padding: int,
    device: str,
    chunk_size: int = 32,
) -> list[np.ndarray]:
    """Extract features for K threshold vectors using two-phase evaluation.

    Phase 1 (expensive, done ONCE per chunk): conv2d accumulation → store
    cumulative potentials at all T time steps.
    Phase 2 (cheap, done for ALL K configs): scan potentials for threshold
    crossings → spike times → features.

    :param thresholds_2d: (K, F) threshold matrix.
    :returns: list of K arrays, each (N, flat_dim).
    """
    N = len(images)
    K, num_filters = thresholds_2d.shape
    kH = weights_4d.shape[2]
    oH = (images.shape[2] + 2 * padding - kH) // stride + 1
    oW = (images.shape[3] + 2 * padding - kH) // stride + 1
    flat_dim = num_filters * pool_size * pool_size

    thresholds_2d = thresholds_2d.to(device)
    w = weights_4d.to(device)

    X_all = [np.empty((N, flat_dim), dtype=np.float32) for _ in range(K)]
    n_chunks = (N + chunk_size - 1) // chunk_size
    t0 = time.time()

    for chunk_idx, start in enumerate(range(0, N, chunk_size)):
        end = min(start + chunk_size, N)
        inp = images[start:end].to(device)
        B = end - start

        # --- Phase 1: Precompute membrane potentials (conv2d, ONCE) ---
        finite_mask = torch.isfinite(inp)
        if not finite_mask.any():
            # No spikes in this chunk — all configs get inf spike times
            for i in range(K):
                feat = spike_times_to_features(
                    torch.full((B, num_filters, oH, oW), float("inf")),
                    t_target=t_target,
                )
                pooled = sum_pool_features(feat, pool_size)
                X_all[i][start:end] = pooled.flatten(1).numpy()
            continue

        unique_times = inp[finite_mask].unique().sort()[0]
        T = len(unique_times)

        # Store cum_potential at each time step: (T, B, F, oH, oW)
        all_potentials = torch.empty(
            T,
            B,
            num_filters,
            oH,
            oW,
            dtype=inp.dtype,
            device=device,
        )
        cum = torch.zeros(B, num_filters, oH, oW, dtype=inp.dtype, device=device)
        for k in range(T):
            active = (inp == unique_times[k]).float()
            contrib = F.conv2d(active, w, stride=stride, padding=padding)
            cum = cum + contrib
            all_potentials[k] = cum

        # --- Phase 2: Find crossings for ALL K configs (cheap comparisons) ---
        thresholds_5d = thresholds_2d.view(K, 1, num_filters, 1, 1)
        result = torch.full(
            (K, B, num_filters, oH, oW),
            float("inf"),
            dtype=inp.dtype,
            device=device,
        )
        not_yet_spiked = torch.ones(
            (K, B, num_filters, oH, oW),
            dtype=torch.bool,
            device=device,
        )

        for k in range(T):
            pot = all_potentials[k]  # (B, F, oH, oW)
            crossed = (pot.unsqueeze(0) >= thresholds_5d) & not_yet_spiked
            result[crossed] = unique_times[k]
            not_yet_spiked &= ~crossed
            if not not_yet_spiked.any():
                break

        del all_potentials, cum  # free GPU memory

        # Convert spike times → features for each config
        for i in range(K):
            feat = spike_times_to_features(result[i].cpu(), t_target=t_target)
            pooled = sum_pool_features(feat, pool_size)
            X_all[i][start:end] = pooled.flatten(1).numpy()

        del result, not_yet_spiked

        if (chunk_idx + 1) % 50 == 0 or chunk_idx == n_chunks - 1:
            elapsed = time.time() - t0
            rate = (chunk_idx + 1) / elapsed
            eta = (n_chunks - chunk_idx - 1) / rate if rate > 0 else 0
            logger.info(
                "    chunk %d/%d (%.0fs elapsed, ETA %.0fs)",
                chunk_idx + 1,
                n_chunks,
                elapsed,
                eta,
            )

    return X_all


# ---------------------------------------------------------------------------
# Classifier helpers
# ---------------------------------------------------------------------------


def make_classifier(name: str, alpha: float = 1.0):
    """Create a classifier by name."""
    if name == "ridge":
        from spiking.evaluation.ridge_column_swap import RidgeColumnSwap

        return RidgeColumnSwap(alpha=alpha)
    if name == "svc":
        from spiking.evaluation.torch_svc import TorchLinearSVC

        return TorchLinearSVC(C=1.0)
    raise ValueError(f"Unknown classifier: {name}")


def evaluate_with_classifier(
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """Fit classifier and return (train_acc, val_acc)."""
    clf.fit(X_train, y_train)
    train_pred = np.asarray(clf.predict(X_train))
    val_pred = np.asarray(clf.predict(X_test))
    return (train_pred == y_train).mean(), (val_pred == y_test).mean()


# ---------------------------------------------------------------------------
# Response surface fitting
# ---------------------------------------------------------------------------


def fit_response_surface(
    design: np.ndarray,
    accuracies: np.ndarray,
    fit_interactions: bool = False,
    alpha: float = 1.0,
) -> dict:
    """Fit a polynomial response surface to the experimental data.

    Model: acc ≈ β₀ + Σ βᵢ dᵢ + [Σ βᵢⱼ dᵢ dⱼ]

    :param design: (N, p) design matrix of ±1 perturbation codes.
    :param accuracies: (N,) measured accuracies.
    :param fit_interactions: include pairwise interaction terms.
    :param alpha: Ridge regularization for the fit.
    :returns: dict with coefficients, predictions, R², etc.
    """
    N, p = design.shape

    # Build regressor matrix
    if fit_interactions:
        # Main effects + pairwise interactions
        n_interact = p * (p - 1) // 2
        X = np.zeros((N, 1 + p + n_interact), dtype=np.float32)
        X[:, 0] = 1.0  # intercept
        X[:, 1 : 1 + p] = design
        col = 1 + p
        for i in range(p):
            for j in range(i + 1, p):
                X[:, col] = design[:, i] * design[:, j]
                col += 1
    else:
        # Main effects only
        X = np.ones((N, 1 + p), dtype=np.float32)
        X[:, 1:] = design

    # Ridge regression fit
    XtX = X.T @ X + alpha * np.eye(X.shape[1], dtype=np.float32)
    Xty = X.T @ accuracies
    beta = np.linalg.solve(XtX, Xty)

    # Predictions and R²
    y_pred = X @ beta
    ss_res = ((accuracies - y_pred) ** 2).sum()
    ss_tot = ((accuracies - accuracies.mean()) ** 2).sum()
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Extract coefficients
    intercept = beta[0]
    main_effects = beta[1 : 1 + p]

    interactions = None
    if fit_interactions:
        interactions = beta[1 + p :]

    return {
        "intercept": float(intercept),
        "main_effects": main_effects,  # (p,)
        "interactions": interactions,
        "r_squared": float(r_squared),
        "residual_std": float(np.sqrt(ss_res / N)),
        "y_pred": y_pred,
    }


def predict_optimal_deltas(surface: dict, scale: float) -> np.ndarray:
    """Predict optimal perturbation direction from the fitted surface.

    For a linear model (no interactions), the optimal δᵢ is simply:
    +scale if βᵢ > 0 (increasing this neuron's threshold helps)
    -scale if βᵢ < 0 (decreasing helps)

    :param surface: output of fit_response_surface.
    :param scale: perturbation magnitude.
    :returns: (p,) optimal delta fractions.
    """
    betas = surface["main_effects"]
    # Continuous optimum: move in direction of gradient, clamped to ±scale
    direction = np.sign(betas)
    return direction * scale


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_rsm_results(
    surface: dict,
    accuracies: np.ndarray,
    baseline_acc: float,
    optimal_acc: float,
    output_path: str,
) -> None:
    """Plot RSM diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. Predicted vs actual accuracy
    ax = axes[0, 0]
    ax.scatter(surface["y_pred"], accuracies, alpha=0.3, s=10)
    lims = [
        min(accuracies.min(), surface["y_pred"].min()),
        max(accuracies.max(), surface["y_pred"].max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Predicted accuracy")
    ax.set_ylabel("Actual accuracy")
    ax.set_title(f"RSM fit (R² = {surface['r_squared']:.4f})")
    ax.grid(True, alpha=0.3)

    # 2. Main effects (top 30)
    ax = axes[0, 1]
    betas = surface["main_effects"]
    top_idx = np.argsort(np.abs(betas))[-30:][::-1]
    colors = ["green" if b > 0 else "red" for b in betas[top_idx]]
    ax.barh(range(len(top_idx)), betas[top_idx], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([f"n{i}" for i in top_idx], fontsize=7)
    ax.set_xlabel("βᵢ (main effect)")
    ax.set_title("Top 30 neurons by |effect|")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # 3. Distribution of accuracies
    ax = axes[1, 0]
    ax.hist(accuracies, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(
        baseline_acc,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Baseline: {baseline_acc:.4f}",
    )
    ax.axvline(
        optimal_acc,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"RSM optimal: {optimal_acc:.4f}",
    )
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of evaluated configs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Effect magnitude distribution
    ax = axes[1, 1]
    ax.hist(np.abs(betas), bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(
        np.abs(betas).mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {np.abs(betas).mean():.6f}",
    )
    n_significant = (np.abs(betas) > 2 * surface["residual_std"]).sum()
    ax.set_xlabel("|βᵢ|")
    ax.set_ylabel("Count")
    ax.set_title(f"Effect magnitudes ({n_significant} significant at 2σ)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info("Plot saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="RSM threshold optimization")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-filters", type=int, default=256)
    parser.add_argument("--t-obj", type=float, default=0.7)
    parser.add_argument("--n-configs", type=int, default=300)
    parser.add_argument("--perturbation-scale", type=float, default=0.05)
    parser.add_argument("--fit-interactions", action="store_true")
    parser.add_argument("--ridge-alpha-fit", type=float, default=1.0)
    parser.add_argument("--classifier", default="ridge", choices=["ridge", "svc"])
    parser.add_argument("--classifier-alpha", type=float, default=1.0)
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--config-batch-size", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    config = RSMConfig(
        **{k: v for k, v in vars(args).items() if k in RSMConfig.__dataclass_fields__}
    )
    set_seed(config.seed)

    model_dir = resolve_model_dir(
        config.dataset, config.num_filters, config.t_obj, config.seed
    )
    model_path = f"{model_dir}/model.pth"
    if not os.path.exists(model_path):
        logger.error("No model at %s", model_path)
        return

    with open(f"{model_dir}/setup.json") as f:
        t_target = json.load(f).get("target_timestamp", config.t_obj)

    output_dir = config.output_dir or f"{model_dir}/rsm_opt"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(asdict(config), f, indent=4)

    # --- Load model + data ---
    logger.info("Loading model from %s", model_path)
    layer = load_model(model_path)

    logger.info("Loading data...")
    train_data, test_data = load_split_data(config.dataset)
    train_images, train_labels = train_data["images"], train_data["labels"]
    test_images, test_labels = test_data["images"], test_data["labels"]
    y_train, y_test = train_labels.numpy(), test_labels.numpy()
    logger.info("Train: %d, Test: %d", len(train_images), len(test_images))

    original_thresholds = layer.thresholds.detach().clone()
    num_filters = config.num_filters
    weights_4d = layer.weights_4d.detach()

    # --- Baseline ---
    logger.info("Computing baseline...")
    baseline_features_train = multi_threshold_features(
        weights_4d,
        original_thresholds.unsqueeze(0),
        train_images,
        t_target,
        config.pool_size,
        layer.stride,
        layer.padding,
        config.device,
        chunk_size=64,
    )[0]
    baseline_features_test = multi_threshold_features(
        weights_4d,
        original_thresholds.unsqueeze(0),
        test_images,
        t_target,
        config.pool_size,
        layer.stride,
        layer.padding,
        config.device,
        chunk_size=64,
    )[0]

    clf = make_classifier(config.classifier, config.classifier_alpha)
    baseline_train_acc, baseline_val_acc = evaluate_with_classifier(
        clf,
        baseline_features_train,
        y_train,
        baseline_features_test,
        y_test,
    )
    logger.info(
        "Baseline %s — train: %.4f, val: %.4f",
        config.classifier,
        baseline_train_acc,
        baseline_val_acc,
    )

    # --- Generate experimental design ---
    logger.info(
        "Generating %d random configurations (scale=%.3f)...",
        config.n_configs,
        config.perturbation_scale,
    )
    design = generate_rademacher_design(config.n_configs, num_filters, config.seed)

    # Convert design to actual thresholds: θ_new = θ_orig * (1 + scale * d_i)
    # design is (N, F) of ±1, scale controls magnitude
    thresholds_all = []
    for i in range(config.n_configs):
        delta_frac = config.perturbation_scale * design[i]  # (F,) of ±scale
        theta_i = original_thresholds * (1.0 + torch.from_numpy(delta_frac))
        thresholds_all.append(theta_i)

    # --- Evaluate all configs (two-phase: conv2d once, threshold check for all) ---
    thresh_tensor = torch.stack(thresholds_all)  # (K, F)
    logger.info(
        "Evaluating %d configs (chunk_size=%d)...",
        config.n_configs,
        config.chunk_size,
    )

    t0 = time.time()
    logger.info("  Extracting train features...")
    X_trains = multi_threshold_features(
        weights_4d,
        thresh_tensor,
        train_images,
        t_target,
        config.pool_size,
        layer.stride,
        layer.padding,
        config.device,
        config.chunk_size,
    )
    logger.info("  Extracting test features...")
    X_tests = multi_threshold_features(
        weights_4d,
        thresh_tensor,
        test_images,
        t_target,
        config.pool_size,
        layer.stride,
        layer.padding,
        config.device,
        config.chunk_size,
    )
    logger.info("  Feature extraction done in %.0fs", time.time() - t0)

    # Fit classifier for each config
    accuracies = np.zeros(config.n_configs, dtype=np.float32)
    train_accs = np.zeros(config.n_configs, dtype=np.float32)
    t1 = time.time()
    for i in range(config.n_configs):
        clf_i = make_classifier(config.classifier, config.classifier_alpha)
        tr_acc, val_acc = evaluate_with_classifier(
            clf_i,
            X_trains[i],
            y_train,
            X_tests[i],
            y_test,
        )
        accuracies[i] = val_acc
        train_accs[i] = tr_acc
    logger.info("  Classifier fitting done in %.0fs", time.time() - t1)
    del X_trains, X_tests

    logger.info("All %d configs evaluated in %.0fs", config.n_configs, time.time() - t0)
    logger.info(
        "Accuracy — mean: %.4f, std: %.4f, min: %.4f, max: %.4f",
        accuracies.mean(),
        accuracies.std(),
        accuracies.min(),
        accuracies.max(),
    )

    # --- Fit response surface ---
    logger.info(
        "Fitting response surface (interactions=%s)...", config.fit_interactions
    )
    surface = fit_response_surface(
        design,
        accuracies,
        fit_interactions=config.fit_interactions,
        alpha=config.ridge_alpha_fit,
    )
    logger.info(
        "R² = %.4f, residual std = %.6f", surface["r_squared"], surface["residual_std"]
    )

    # Significant effects
    betas = surface["main_effects"]
    threshold_2sigma = 2 * surface["residual_std"]
    significant = np.abs(betas) > threshold_2sigma
    n_sig = significant.sum()
    logger.info(
        "%d / %d neurons have significant effects (|β| > 2σ)", n_sig, num_filters
    )

    # Top 10 effects
    top10 = np.argsort(np.abs(betas))[-10:][::-1]
    for idx in top10:
        logger.info(
            "  Neuron %3d: β = %+.6f (%s threshold helps)",
            idx,
            betas[idx],
            "lower" if betas[idx] < 0 else "higher",
        )

    # --- Predict optimal thresholds ---
    optimal_deltas = predict_optimal_deltas(surface, config.perturbation_scale)
    optimal_thresholds = original_thresholds * (
        1.0 + torch.from_numpy(optimal_deltas).float()
    )

    # --- Evaluate optimal thresholds ---
    logger.info("Evaluating RSM-predicted optimal thresholds...")
    X_train_opt = multi_threshold_features(
        weights_4d,
        optimal_thresholds.unsqueeze(0),
        train_images,
        t_target,
        config.pool_size,
        layer.stride,
        layer.padding,
        config.device,
        chunk_size=64,
    )[0]
    X_test_opt = multi_threshold_features(
        weights_4d,
        optimal_thresholds.unsqueeze(0),
        test_images,
        t_target,
        config.pool_size,
        layer.stride,
        layer.padding,
        config.device,
        chunk_size=64,
    )[0]

    clf_opt = make_classifier(config.classifier, config.classifier_alpha)
    opt_train_acc, opt_val_acc = evaluate_with_classifier(
        clf_opt,
        X_train_opt,
        y_train,
        X_test_opt,
        y_test,
    )
    logger.info(
        "Optimal %s — train: %.4f, val: %.4f",
        config.classifier,
        opt_train_acc,
        opt_val_acc,
    )

    # Also find the best config actually evaluated (empirical best)
    best_idx = np.argmax(accuracies)
    best_empirical_acc = accuracies[best_idx]
    logger.info("Best empirical config: #%d — val: %.4f", best_idx, best_empirical_acc)

    # --- Save results ---
    results = {
        "baseline": {
            "train_acc": float(baseline_train_acc),
            "val_acc": float(baseline_val_acc),
        },
        "rsm_optimal": {
            "train_acc": float(opt_train_acc),
            "val_acc": float(opt_val_acc),
        },
        "best_empirical": {
            "index": int(best_idx),
            "val_acc": float(best_empirical_acc),
        },
        "surface": {
            "r_squared": surface["r_squared"],
            "residual_std": surface["residual_std"],
            "n_significant": int(n_sig),
            "main_effects": surface["main_effects"].tolist(),
        },
        "original_thresholds": original_thresholds.tolist(),
        "optimal_thresholds": optimal_thresholds.tolist(),
        "accuracies": accuracies.tolist(),
        "train_accuracies": train_accs.tolist(),
        "config": asdict(config),
    }
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Results saved to %s/results.json", output_dir)

    # Plot
    plot_rsm_results(
        surface,
        accuracies,
        baseline_val_acc,
        opt_val_acc,
        f"{output_dir}/rsm_analysis.png",
    )

    # --- Summary ---
    logger.info("=== Summary ===")
    logger.info("Baseline:      %.4f", baseline_val_acc)
    logger.info(
        "RSM optimal:   %.4f (%+.4f)", opt_val_acc, opt_val_acc - baseline_val_acc
    )
    logger.info(
        "Best empirical: %.4f (%+.4f)",
        best_empirical_acc,
        best_empirical_acc - baseline_val_acc,
    )
    logger.info("R² of surface: %.4f", surface["r_squared"])
