"""Sanity check: STE gradient sign vs brute-force perturbation direction."""

import json
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from applications.common import load_split_data, resolve_model_dir, set_seed
from applications.supervised_threshold_optimization.train import (
    differentiable_features,
    extract_hard_features,
    init_classifier_from_svc,
)
from spiking.utils.checkpoints import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_loss(spike_module, classifier, criterion, loader, t_target, pool_size, device):
    spike_module.eval()
    classifier.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            st, _ = spike_module(images, first_spike_only=False)
            feats = differentiable_features(st, t_target, pool_size)
            logits = classifier(feats)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    set_seed(1)
    device = torch.device("cuda")
    model_dir = resolve_model_dir("cifar10", 256, 0.7, 1)
    layer = load_model(f"{model_dir}/model.pth")

    with open(f"{model_dir}/setup.json") as f:
        t_target = json.load(f).get("target_timestamp", 0.7)

    train_data, test_data = load_split_data("cifar10")
    train_images, train_labels = train_data["images"], train_data["labels"]
    test_images, test_labels = test_data["images"], test_data["labels"]

    logger.info("Computing hard features...")
    X_train = extract_hard_features(layer, train_images, t_target, 2, 2048, "cuda")
    X_test = extract_hard_features(layer, test_images, t_target, 2, 2048, "cuda")
    y_train, y_test = train_labels.numpy(), test_labels.numpy()

    classifier = init_classifier_from_svc(X_train, y_train).to(device)

    layer = layer.to(device)
    layer.eval()
    layer._backend = "differential_dense"
    layer.tau = 0.01
    layer.t_no_spike = 1.0
    layer.weights.requires_grad_(False)
    spike_module = layer

    criterion = nn.MultiMarginLoss()

    val_loader = DataLoader(
        TensorDataset(test_images[:5000], test_labels[:5000]),
        batch_size=64, shuffle=False,
    )
    train_loader = DataLoader(
        TensorDataset(train_images[:5000], train_labels[:5000]),
        batch_size=64, shuffle=False,
    )

    logger.info("=== Test 1: STE features vs hard features ===")
    spike_module.eval()
    ste_feats = []
    with torch.no_grad():
        for images, labels in DataLoader(TensorDataset(test_images[:500], test_labels[:500]), batch_size=64):
            st, _ = spike_module(images.to(device), first_spike_only=False)
            feats = differentiable_features(st, t_target, 2)
            ste_feats.append(feats.cpu().numpy())
    ste_feats = np.concatenate(ste_feats)
    hard_feats = X_test[:500]
    diff = np.abs(ste_feats - hard_feats)
    logger.info("STE vs hard features — max diff: %.6f, mean diff: %.6f", diff.max(), diff.mean())

    with torch.no_grad():
        ste_logits = classifier(torch.tensor(ste_feats, device=device))
        hard_logits = classifier(torch.tensor(hard_feats, device=device))
        ste_acc = (ste_logits.argmax(1).cpu() == test_labels[:500]).float().mean().item()
        hard_acc = (hard_logits.argmax(1).cpu() == test_labels[:500]).float().mean().item()
    logger.info("STE acc: %.4f, Hard acc: %.4f", ste_acc, hard_acc)

    logger.info("=== Test 2: Gradient direction verification ===")

    spike_module.train()
    accumulated_grad = torch.zeros(256, device=device)
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        if spike_module.thresholds.grad is not None:
            spike_module.thresholds.grad.zero_()
        st, _ = spike_module(images, first_spike_only=False)
        feats = differentiable_features(st, t_target, 2)
        logits = classifier(feats)
        loss = criterion(logits, labels)
        loss.backward()
        accumulated_grad += spike_module.thresholds.grad.detach()

    grad_sign = torch.sign(accumulated_grad)
    logger.info("Gradient stats — positive: %d, negative: %d, zero: %d",
                (grad_sign > 0).sum().item(),
                (grad_sign < 0).sum().item(),
                (grad_sign == 0).sum().item())
    logger.info("Gradient magnitude — mean: %.4f, max: %.4f",
                accumulated_grad.abs().mean().item(),
                accumulated_grad.abs().max().item())

    logger.info("=== Test 3: Brute-force perturbation vs gradient ===")

    baseline_loss, baseline_acc = compute_loss(
        spike_module, classifier, criterion, val_loader, t_target, 2, "cuda"
    )
    logger.info("Baseline — loss: %.4f, acc: %.4f", baseline_loss, baseline_acc)

    original_thresholds = spike_module.thresholds.data.clone()

    step = 0.001
    spike_module.thresholds.data = original_thresholds - step * grad_sign
    gd_loss, gd_acc = compute_loss(
        spike_module, classifier, criterion, val_loader, t_target, 2, "cuda"
    )
    logger.info("Gradient descent (all, step=%.4f) — loss: %.4f (%+.4f), acc: %.4f (%+.4f)",
                step, gd_loss, gd_loss - baseline_loss, gd_acc, gd_acc - baseline_acc)

    spike_module.thresholds.data = original_thresholds + step * grad_sign
    ga_loss, ga_acc = compute_loss(
        spike_module, classifier, criterion, val_loader, t_target, 2, "cuda"
    )
    logger.info("Gradient ascent  (all, step=%.4f) — loss: %.4f (%+.4f), acc: %.4f (%+.4f)",
                step, ga_loss, ga_loss - baseline_loss, ga_acc, ga_acc - baseline_acc)

    spike_module.thresholds.data = original_thresholds + step * (2 * torch.randint(0, 2, (256,), device=device).float() - 1)
    rand_loss, rand_acc = compute_loss(
        spike_module, classifier, criterion, val_loader, t_target, 2, "cuda"
    )
    logger.info("Random           (all, step=%.4f) — loss: %.4f (%+.4f), acc: %.4f (%+.4f)",
                step, rand_loss, rand_loss - baseline_loss, rand_acc, rand_acc - baseline_acc)

    spike_module.thresholds.data = original_thresholds

    logger.info("=== Test 4: Per-neuron perturbation (top 10) ===")
    top_neurons = accumulated_grad.abs().topk(10).indices
    step = 0.01

    for idx in top_neurons:
        i = idx.item()
        g = accumulated_grad[i].item()

        spike_module.thresholds.data = original_thresholds.clone()
        spike_module.thresholds.data[i] -= step * np.sign(g)
        gd_loss_i, gd_acc_i = compute_loss(
            spike_module, classifier, criterion, val_loader, t_target, 2, "cuda"
        )

        spike_module.thresholds.data = original_thresholds.clone()
        spike_module.thresholds.data[i] += step * np.sign(g)
        ga_loss_i, ga_acc_i = compute_loss(
            spike_module, classifier, criterion, val_loader, t_target, 2, "cuda"
        )

        logger.info(
            "Neuron %3d | grad=%+8.3f | descent: loss=%+.4f acc=%+.4f | ascent: loss=%+.4f acc=%+.4f",
            i, g,
            gd_loss_i - baseline_loss, gd_acc_i - baseline_acc,
            ga_loss_i - baseline_loss, ga_acc_i - baseline_acc,
        )

    spike_module.thresholds.data = original_thresholds


if __name__ == "__main__":
    main()
