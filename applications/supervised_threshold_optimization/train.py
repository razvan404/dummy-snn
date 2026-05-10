"""Supervised threshold optimization via differentiable spike times."""

import argparse
import json
import logging
import math
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from applications.common import load_split_data, resolve_model_dir, set_seed
from spiking.evaluation.conv_feature_extraction import sum_pool_features
from spiking.evaluation.ridge_column_swap import RidgeColumnSwap
from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer
from spiking.utils.checkpoints import load_model

logger = logging.getLogger(__name__)


@dataclass
class Config:
    dataset: str = "cifar10"
    seed: int = 1
    num_filters: int = 256
    t_obj: float = 0.7

    lr_threshold: float = 1e-3
    lr_classifier: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    batch_size: int = 64
    epochs: int = 50
    warmup_epochs: int = 0
    loss: str = "hinge"
    classifier_init: str = "svc"

    sign_update: bool = False
    sign_step_size: float = 0.01

    tau_start: float = 0.1
    tau_end: float = 0.05
    tau_schedule: str = "cosine"

    pool_size: int = 2
    ridge_alpha: float = 1.0

    device: str = "cpu"
    output_dir: str = ""


def get_tau(config: Config, epoch: int) -> float:
    if config.epochs <= 1:
        return config.tau_end
    frac = epoch / (config.epochs - 1)
    if config.tau_schedule == "cosine":
        return config.tau_end + 0.5 * (config.tau_start - config.tau_end) * (
            1 + math.cos(math.pi * frac)
        )
    return config.tau_start + frac * (config.tau_end - config.tau_start)


def differentiable_features(
    soft_spike_times: torch.Tensor,
    t_target: float,
    pool_size: int,
) -> torch.Tensor:
    """Soft spike times → pooled features; clamp matches the hard pipeline."""
    features = 1.0 - (soft_spike_times - t_target) / (1.0 - t_target)
    features = features.clamp(0.0, 1.0)
    pooled = sum_pool_features(features, pool_size)
    return pooled.flatten(1)


def extract_hard_features(
    layer,
    images: torch.Tensor,
    t_target: float,
    pool_size: int,
    chunk_size: int,
    device: str,
) -> np.ndarray:
    from spiking.evaluation.conv_feature_extraction import sum_pool_features as sp
    from spiking.evaluation.feature_extraction import spike_times_to_features

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
            pooled = sp(feat, pool_size)
            X[start:end] = pooled.flatten(1).numpy()

    layer.cpu()
    return X


def init_classifier_from_ridge(
    ridge: RidgeColumnSwap,
    in_features: int,
    num_classes: int,
) -> nn.Linear:
    linear = nn.Linear(in_features, num_classes)
    w = ridge.weights
    intercept = ridge._to_np(ridge._intercept)
    linear.weight.data = torch.tensor(w.T, dtype=torch.float32)
    linear.bias.data = torch.tensor(intercept, dtype=torch.float32).squeeze()
    return linear


def init_classifier_from_svc(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> nn.Linear:
    """Fit LinearSVC (cuml if available, else sklearn) and transfer to ``nn.Linear``."""
    try:
        from cuml.svm import LinearSVC

        logger.info("Fitting LinearSVC (cuml GPU) for classifier initialization...")
        svc = LinearSVC(tol=1e-3, max_iter=10000)
    except ImportError:
        from sklearn.svm import LinearSVC

        logger.info("Fitting LinearSVC (sklearn CPU) for classifier initialization...")
        svc = LinearSVC(max_iter=5000, dual="auto")

    svc.fit(X_train, y_train)
    train_acc = (np.asarray(svc.predict(X_train)) == y_train).mean()
    logger.info("SVC init — train acc: %.4f", train_acc)

    coef = np.asarray(svc.coef_)
    intercept = np.asarray(svc.intercept_)
    num_classes, in_features = coef.shape
    linear = nn.Linear(in_features, num_classes)
    linear.weight.data = torch.tensor(coef, dtype=torch.float32)
    linear.bias.data = torch.tensor(intercept, dtype=torch.float32)
    return linear


def get_criterion(loss_name: str) -> nn.Module:
    if loss_name == "hinge":
        return nn.MultiMarginLoss()
    return nn.CrossEntropyLoss()


def evaluate_epoch(
    spike_module: ConvIntegrateAndFireLayer,
    classifier: nn.Linear,
    criterion: nn.Module,
    loader: DataLoader,
    t_target: float,
    pool_size: int,
    device: str,
) -> tuple[float, float]:
    spike_module.eval()
    classifier.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            soft_st, _ = spike_module(images, first_spike_only=False)
            feats = differentiable_features(soft_st, t_target, pool_size)
            logits = classifier(feats)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return correct / total, total_loss / total


def train_epoch(
    spike_module: ConvIntegrateAndFireLayer,
    classifier: nn.Linear,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    t_target: float,
    pool_size: int,
    device: str,
    grad_clip: float = 1.0,
    freeze_thresholds: bool = False,
) -> tuple[float, float]:
    spike_module.train()
    classifier.train()

    if freeze_thresholds:
        spike_module.thresholds.requires_grad_(False)
    else:
        spike_module.thresholds.requires_grad_(True)

    correct = 0
    total = 0
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        soft_st, _ = spike_module(images, first_spike_only=False)
        feats = differentiable_features(soft_st, t_target, pool_size)
        logits = classifier(feats)
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0 and not freeze_thresholds:
            nn.utils.clip_grad_norm_([spike_module.thresholds], grad_clip)

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return correct / total, total_loss / total


def train_epoch_sign(
    spike_module: ConvIntegrateAndFireLayer,
    classifier: nn.Linear,
    criterion: nn.Module,
    loader: DataLoader,
    t_target: float,
    pool_size: int,
    device: str,
    step_size: float = 0.01,
) -> tuple[float, float]:
    """Sign-based threshold update: θ ← θ − step·sign(Σ grad). Classifier frozen."""
    spike_module.train()
    classifier.eval()
    spike_module.thresholds.requires_grad_(True)

    accumulated_grad = torch.zeros_like(spike_module.thresholds)
    correct = 0
    total = 0
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if spike_module.thresholds.grad is not None:
            spike_module.thresholds.grad.zero_()

        st, _ = spike_module(images, first_spike_only=False)
        feats = differentiable_features(st, t_target, pool_size)
        logits = classifier(feats)
        loss = criterion(logits, labels)
        loss.backward()

        accumulated_grad += spike_module.thresholds.grad.detach()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    with torch.no_grad():
        spike_module.thresholds -= step_size * torch.sign(accumulated_grad)

    return correct / total, total_loss / total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supervised threshold optimization via differentiable spike times"
    )
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-filters", type=int, default=256)
    parser.add_argument("--t-obj", type=float, default=0.7)
    parser.add_argument("--lr-threshold", type=float, default=1e-3)
    parser.add_argument("--lr-classifier", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--loss", default="hinge", choices=["hinge", "ce"])
    parser.add_argument(
        "--classifier-init", default="svc", choices=["svc", "ridge", "random"]
    )
    parser.add_argument(
        "--sign-update",
        action="store_true",
        help="Sign-based updates: fixed step in gradient direction per epoch",
    )
    parser.add_argument("--sign-step-size", type=float, default=0.01)
    parser.add_argument("--tau-start", type=float, default=0.1)
    parser.add_argument("--tau-end", type=float, default=0.05)
    parser.add_argument(
        "--tau-schedule", default="cosine", choices=["cosine", "linear"]
    )
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    raw = vars(args)
    config = Config(
        **{k: v for k, v in raw.items() if k in Config.__dataclass_fields__}
    )

    set_seed(config.seed)
    device = torch.device(config.device)

    model_dir = resolve_model_dir(
        config.dataset, config.num_filters, config.t_obj, config.seed
    )
    model_path = f"{model_dir}/model.pth"
    if not os.path.exists(model_path):
        logger.error("No model at %s", model_path)
        return

    with open(f"{model_dir}/setup.json") as f:
        setup = json.load(f)
    t_target = setup.get("target_timestamp", config.t_obj)

    output_dir = config.output_dir or f"{model_dir}/supervised_opt"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(asdict(config), f, indent=4)

    logger.info("Loading model from %s", model_path)
    layer = load_model(model_path)

    logger.info("Loading data...")
    train_data, test_data = load_split_data(config.dataset)
    train_images, train_labels = train_data["images"], train_data["labels"]
    test_images, test_labels = test_data["images"], test_data["labels"]
    logger.info(
        "Train: %d images, Test: %d images", len(train_images), len(test_images)
    )

    logger.info("Computing baseline features...")
    X_train = extract_hard_features(
        layer,
        train_images,
        t_target,
        config.pool_size,
        chunk_size=2048,
        device=config.device,
    )
    X_test = extract_hard_features(
        layer,
        test_images,
        t_target,
        config.pool_size,
        chunk_size=2048,
        device=config.device,
    )
    y_train = train_labels.numpy()
    y_test = test_labels.numpy()

    logger.info("Fitting Ridge baseline...")
    ridge = RidgeColumnSwap(alpha=config.ridge_alpha)
    ridge.fit(X_train, y_train)
    baseline_ridge_train = (ridge.predict(X_train) == y_train).mean()
    baseline_ridge_val = (ridge.predict(X_test) == y_test).mean()
    logger.info(
        "Baseline Ridge — train: %.4f, val: %.4f",
        baseline_ridge_train,
        baseline_ridge_val,
    )

    from spiking.evaluation import evaluate_classifier

    baseline_svc_train, baseline_svc_val = evaluate_classifier(
        X_train, y_train, X_test, y_test
    )
    logger.info(
        "Baseline SVC   — train: %.4f, val: %.4f",
        baseline_svc_train["accuracy"],
        baseline_svc_val["accuracy"],
    )

    original_thresholds = layer.thresholds.detach().clone()
    layer = layer.to(device)
    layer.eval()
    layer._backend = "differential_dense"
    layer.tau = config.tau_start
    layer.t_no_spike = 1.0
    layer.weights.requires_grad_(False)
    spike_module = layer

    in_features = config.num_filters * config.pool_size * config.pool_size
    num_classes = len(np.unique(y_train))

    if config.classifier_init == "svc":
        classifier = init_classifier_from_svc(X_train, y_train).to(device)
    elif config.classifier_init == "ridge":
        classifier = init_classifier_from_ridge(ridge, in_features, num_classes).to(
            device
        )
    else:
        classifier = nn.Linear(in_features, num_classes).to(device)
    del ridge

    criterion = get_criterion(config.loss)
    mode = "STE+sign" if config.sign_update else "STE"
    logger.info(
        "Loss: %s, Classifier init: %s, Mode: %s",
        config.loss,
        config.classifier_init,
        mode,
    )

    optimizer = torch.optim.Adam(
        [
            {"params": [spike_module.thresholds], "lr": config.lr_threshold},
            {"params": classifier.parameters(), "lr": config.lr_classifier},
        ],
        weight_decay=config.weight_decay,
    )

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    history = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
        "tau": [],
        "threshold_drift": [],
    }
    best_val_acc = 0.0

    logger.info(
        "Starting training: %d warmup + %d joint epochs...",
        config.warmup_epochs,
        config.epochs,
    )
    total_epochs = config.warmup_epochs + config.epochs
    for epoch in range(total_epochs):
        is_warmup = epoch < config.warmup_epochs
        opt_epoch = max(0, epoch - config.warmup_epochs)
        tau = get_tau(config, opt_epoch) if not is_warmup else config.tau_start
        spike_module.tau = tau

        if config.sign_update and not is_warmup:
            train_acc, train_loss = train_epoch_sign(
                spike_module,
                classifier,
                criterion,
                train_loader,
                t_target,
                config.pool_size,
                config.device,
                step_size=config.sign_step_size,
            )
        else:
            train_acc, train_loss = train_epoch(
                spike_module,
                classifier,
                criterion,
                optimizer,
                train_loader,
                t_target,
                config.pool_size,
                config.device,
                grad_clip=config.grad_clip,
                freeze_thresholds=is_warmup,
            )

        val_acc, val_loss = evaluate_epoch(
            spike_module,
            classifier,
            criterion,
            val_loader,
            t_target,
            config.pool_size,
            config.device,
        )

        threshold_drift = (
            (spike_module.thresholds.detach().cpu() - original_thresholds).norm().item()
        )

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["tau"].append(tau)
        history["threshold_drift"].append(threshold_drift)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "thresholds": spike_module.thresholds.detach().cpu(),
                    "classifier_weight": classifier.weight.detach().cpu(),
                    "classifier_bias": classifier.bias.detach().cpu(),
                },
                f"{output_dir}/best_checkpoint.pt",
            )

        phase = "warmup" if is_warmup else "joint "
        logger.info(
            "[%s] Epoch %3d/%d | τ=%.3f | train: %.4f (%.4f) | val: %.4f (%.4f) | drift: %.3f",
            phase,
            epoch + 1,
            total_epochs,
            tau,
            train_acc,
            train_loss,
            val_acc,
            val_loss,
            threshold_drift,
        )

    logger.info("=== Final Results ===")

    best = torch.load(f"{output_dir}/best_checkpoint.pt", weights_only=True)
    spike_module.thresholds.data = best["thresholds"].to(device)
    classifier.weight.data = best["classifier_weight"].to(device)
    classifier.bias.data = best["classifier_bias"].to(device)

    final_val_acc, final_val_loss = evaluate_epoch(
        spike_module,
        classifier,
        criterion,
        val_loader,
        t_target,
        config.pool_size,
        config.device,
    )
    logger.info("Best supervised — val: %.4f", final_val_acc)

    logger.info("Evaluating optimized thresholds with fresh classifiers...")
    layer.thresholds.data = best["thresholds"]
    X_train_opt = extract_hard_features(
        layer,
        train_images,
        t_target,
        config.pool_size,
        chunk_size=2048,
        device=config.device,
    )
    X_test_opt = extract_hard_features(
        layer,
        test_images,
        t_target,
        config.pool_size,
        chunk_size=2048,
        device=config.device,
    )

    ridge_opt = RidgeColumnSwap(alpha=config.ridge_alpha)
    ridge_opt.fit(X_train_opt, y_train)
    opt_ridge_train = (ridge_opt.predict(X_train_opt) == y_train).mean()
    opt_ridge_val = (ridge_opt.predict(X_test_opt) == y_test).mean()
    logger.info(
        "Optimized thresholds + Ridge — train: %.4f, val: %.4f",
        opt_ridge_train,
        opt_ridge_val,
    )

    opt_svc_train, opt_svc_val = evaluate_classifier(
        X_train_opt, y_train, X_test_opt, y_test
    )
    logger.info(
        "Optimized thresholds + SVC   — train: %.4f, val: %.4f",
        opt_svc_train["accuracy"],
        opt_svc_val["accuracy"],
    )

    results = {
        "baseline": {
            "ridge_train_acc": float(baseline_ridge_train),
            "ridge_val_acc": float(baseline_ridge_val),
            "svc_train_acc": float(baseline_svc_train["accuracy"]),
            "svc_val_acc": float(baseline_svc_val["accuracy"]),
        },
        "supervised": {
            "best_val_acc": float(best_val_acc),
            "final_val_acc": float(final_val_acc),
            "final_val_loss": float(final_val_loss),
        },
        "optimized_thresholds": {
            "ridge_train_acc": float(opt_ridge_train),
            "ridge_val_acc": float(opt_ridge_val),
            "svc_train_acc": float(opt_svc_train["accuracy"]),
            "svc_val_acc": float(opt_svc_val["accuracy"]),
        },
        "original_thresholds": original_thresholds.tolist(),
        "optimized_thresholds_values": best["thresholds"].tolist(),
        "history": history,
    }
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Results saved to %s/results.json", output_dir)
    logger.info("=== Summary ===")
    logger.info(
        "Ridge: %.4f → %.4f (%+.4f)",
        baseline_ridge_val,
        opt_ridge_val,
        opt_ridge_val - baseline_ridge_val,
    )
    logger.info(
        "SVC:   %.4f → %.4f (%+.4f)",
        baseline_svc_val["accuracy"],
        opt_svc_val["accuracy"],
        opt_svc_val["accuracy"] - baseline_svc_val["accuracy"],
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
