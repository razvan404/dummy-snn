"""Train a bounded-encoder ANN classifier on CIFAR-10.

Encoder weights are clipped to [0, 1] and pulled toward {0, 1} by the same
``min(w**2, (w-1)**2)`` penalty as the autoencoder experiment. The rest of
the network (BatchNorm + adaptive max-pool to 2×2 + linear head) is free.
This tests whether the bimodal weight pattern still emerges when the
downstream task is *classification* rather than reconstruction.
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset

from applications.ann_bimodal_autoencoder.model import (
    bimodal_penalty,
    bimodality_score,
)
from applications.ann_bimodal_autoencoder.visualize import (
    save_bimodality_curve,
    save_filter_grid,
    save_filter_grid_whitened,
    save_weight_histogram,
)
from applications.ann_bimodal_classifier.model import ConvBimodalClassifier
from applications.common import set_seed
from applications.datasets.cifar10_whitened import Cifar10WhitenedDataset

logger = logging.getLogger(__name__)


INPUT_MODES = ("cifar10", "cifar10_whitened")
_INPUT_CHANNELS = {"cifar10": 3, "cifar10_whitened": 6}


@dataclass
class Config:
    seed: int = 1
    input_mode: str = "cifar10_whitened"
    data_root: str = "data"

    num_filters: int = 256
    kernel_size: int = 5
    pool_out: int = 2
    num_classes: int = 10

    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    lambda_bimodal: float = 0.1
    train_subset: int = 0

    device: str = "cpu"
    output_dir: str = ""
    num_workers: int = 2


def _resolve_output_dir(config: Config) -> Path:
    if config.output_dir:
        return Path(config.output_dir)
    return Path(
        f"logs/ann_bimodal_classifier/{config.input_mode}/"
        f"seed_{config.seed}/lambda_{config.lambda_bimodal:g}"
    )


def _raw_cifar10(data_root: str, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ds = torchvision.datasets.CIFAR10(root=data_root, train=train, download=True)
    images = torch.from_numpy(ds.data).float() / 255.0
    images = images.permute(0, 3, 1, 2).contiguous()
    labels = torch.tensor(ds.targets, dtype=torch.long)
    return images, labels


class _RgbClassDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


class _WhitenedClassDataset(Dataset):
    """Yields (whitened_intensity_6ch, label)."""

    def __init__(self, whitened: Cifar10WhitenedDataset):
        self._w = whitened

    def __len__(self) -> int:
        return len(self._w)

    def __getitem__(self, idx: int):
        times = self._w.all_times[idx]
        finite = torch.isfinite(times)
        intensity = torch.zeros_like(times)
        intensity[finite] = (1.0 - times[finite]).clamp_(0.0, 1.0)
        return intensity, self._w.outputs[idx]


def _make_loaders(config: Config) -> tuple[DataLoader, DataLoader]:
    if config.input_mode == "cifar10":
        train_imgs, train_labels = _raw_cifar10(config.data_root, train=True)
        test_imgs, test_labels = _raw_cifar10(config.data_root, train=False)
        train_ds: Dataset = _RgbClassDataset(train_imgs, train_labels)
        test_ds: Dataset = _RgbClassDataset(test_imgs, test_labels)
    elif config.input_mode == "cifar10_whitened":
        train_w = Cifar10WhitenedDataset(config.data_root, "train")
        test_w = Cifar10WhitenedDataset(
            config.data_root, "test", kernels=train_w.kernels, mean=train_w.mean
        )
        train_ds = _WhitenedClassDataset(train_w)
        test_ds = _WhitenedClassDataset(test_w)
    else:
        raise ValueError(f"unknown input_mode: {config.input_mode!r}")

    if config.train_subset > 0:
        train_ds = Subset(train_ds, range(min(config.train_subset, len(train_ds))))

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=(config.device != "cpu"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(config.device != "cpu"),
    )
    return train_loader, test_loader


def _evaluate(
    model: ConvBimodalClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n += labels.size(0)
    return correct / max(n, 1), total_loss / max(n, 1)


def _train_epoch(
    model: ConvBimodalClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    lambda_bimodal: float,
    device: str,
) -> tuple[float, float, float, float]:
    model.train()
    total_ce = 0.0
    total_pen = 0.0
    total_loss = 0.0
    correct = 0
    n = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        ce = criterion(logits, labels)
        pen = bimodal_penalty(model.encoder.weight)
        loss = ce + lambda_bimodal * pen

        loss.backward()
        optimizer.step()
        model.clip_encoder_weights()

        bs = labels.size(0)
        total_ce += ce.item() * bs
        total_pen += pen.item() * bs
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == labels).sum().item()
        n += bs

    return (
        correct / n,
        total_ce / n,
        total_pen / n,
        total_loss / n,
    )


def _save_acc_curve(history: dict, path: Path) -> None:
    epochs = np.arange(1, len(history["train_acc"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_acc"], color="C0", label="train")
    ax.plot(epochs, history["val_acc"], color="C1", label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Classification accuracy")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_loss_curve(history: dict, path: Path) -> None:
    epochs = np.arange(1, len(history["ce_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, history["ce_loss"], color="C0", label="cross-entropy")
    ax1.plot(epochs, history["total_loss"], color="C2", label="total")
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
    plt.close()


def _run_one(config: Config) -> dict:
    set_seed(config.seed)
    out_dir = _resolve_output_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=4)

    train_loader, test_loader = _make_loaders(config)
    device = torch.device(config.device)

    in_channels = _INPUT_CHANNELS[config.input_mode]
    model = ConvBimodalClassifier(
        in_channels=in_channels,
        num_filters=config.num_filters,
        kernel_size=config.kernel_size,
        num_classes=config.num_classes,
        pool_out=config.pool_out,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    save_weight_histogram(
        model.encoder.weight,
        out_dir / "weight_hist_epoch0.png",
        title=f"Encoder weights — init (λ={config.lambda_bimodal})",
    )

    history = {
        "train_acc": [],
        "val_acc": [],
        "ce_loss": [],
        "bimodal_loss": [],
        "total_loss": [],
        "bimodality_score": [],
    }
    mid_epoch = max(1, config.epochs // 2)

    for epoch in range(1, config.epochs + 1):
        train_acc, ce, pen, total = _train_epoch(
            model, train_loader, criterion, optimizer, config.lambda_bimodal, device
        )
        val_acc, _ = _evaluate(model, test_loader, criterion, device)
        score = bimodality_score(model.encoder.weight)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["ce_loss"].append(ce)
        history["bimodal_loss"].append(pen)
        history["total_loss"].append(total)
        history["bimodality_score"].append(score)

        logger.info(
            "epoch %3d/%d | train_acc %.4f | val_acc %.4f | ce %.4f | pen %.4f | bimod %.3f",
            epoch,
            config.epochs,
            train_acc,
            val_acc,
            ce,
            pen,
            score,
        )

        if epoch == mid_epoch:
            save_weight_histogram(
                model.encoder.weight,
                out_dir / "weight_hist_epoch_mid.png",
                title=f"Encoder weights — epoch {epoch} (λ={config.lambda_bimodal})",
            )

    save_weight_histogram(
        model.encoder.weight,
        out_dir / "weight_hist_final.png",
        title=f"Encoder weights — final (λ={config.lambda_bimodal})",
    )
    if config.input_mode == "cifar10_whitened":
        save_filter_grid_whitened(
            model.encoder.weight, out_dir / "filter_grid_final.png"
        )
    else:
        save_filter_grid(
            model.encoder.weight,
            out_dir / "filter_grid_final.png",
            rescale_per_filter=True,
        )
    save_bimodality_curve(history, out_dir / "bimodality_curve.png")
    _save_acc_curve(history, out_dir / "accuracy_curve.png")
    _save_loss_curve(history, out_dir / "loss_curves.png")

    final_val_acc, final_val_loss = _evaluate(model, test_loader, criterion, device)

    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "norm": model.norm.state_dict(),
            "classifier": model.classifier.state_dict(),
        },
        out_dir / "checkpoint.pt",
    )

    summary = {
        "config": asdict(config),
        "final_val_acc": float(final_val_acc),
        "final_val_loss": float(final_val_loss),
        "final_bimodality": float(history["bimodality_score"][-1]),
        "final_bimodal_penalty": float(history["bimodal_loss"][-1]),
        "history": history,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=4)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=4)

    logger.info(
        "done — val_acc %.4f | bimodality %.3f → %s",
        final_val_acc,
        history["bimodality_score"][-1],
        out_dir,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--input-mode", default="cifar10_whitened", choices=list(INPUT_MODES)
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--num-filters", type=int, default=256)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--pool-out", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-bimodal", type=float, default=0.1)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    cfg = Config(
        **{k: v for k, v in vars(args).items() if k in Config.__dataclass_fields__}
    )
    _run_one(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
