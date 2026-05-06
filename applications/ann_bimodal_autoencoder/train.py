"""Train the bounded-encoder ANN autoencoder with a bimodal weight regularizer.

Tests whether STDP-style bimodal weights (peaks near 0 and 1) emerge in a
plain ANN when (a) encoder weights are clipped to [0, 1] and (b) a smooth
bimodal penalty is added to the reconstruction loss. Decoder is unconstrained.
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset

from applications.ann_bimodal_autoencoder.model import (
    ConvAutoencoder,
    bimodal_penalty,
    bimodality_score,
)
from applications.ann_bimodal_autoencoder.visualize import (
    save_bimodality_curve,
    save_filter_grid,
    save_filter_grid_whitened,
    save_lambda_sweep,
    save_loss_curves,
    save_reconstructions,
    save_weight_histogram,
)
from applications.common import set_seed
from applications.datasets.cifar10_whitened import Cifar10WhitenedDataset

logger = logging.getLogger(__name__)


INPUT_MODES = ("cifar10", "cifar10_whitened")


@dataclass
class Config:
    seed: int = 1
    dataset: str = "cifar10"  # legacy field, kept for run-dir naming
    input_mode: str = (
        "cifar10"  # "cifar10" (3ch RGB→RGB) or "cifar10_whitened" (6ch→RGB)
    )
    data_root: str = "data"

    num_filters: int = 256
    kernel_size: int = 5

    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    lambda_bimodal: float = 0.1
    train_subset: int = 0  # 0 = use all training images

    device: str = "cpu"
    output_dir: str = ""
    num_workers: int = 2


def _resolve_output_dir(config: Config) -> Path:
    if config.output_dir:
        return Path(config.output_dir)
    return Path(
        f"logs/ann_bimodal_autoencoder/{config.input_mode}/"
        f"seed_{config.seed}/lambda_{config.lambda_bimodal:g}"
    )


class _WhitenedToRgbDataset(Dataset):
    """Yields (whitened_intensity_6ch, raw_rgb_3ch) pairs.

    The whitened spike-time tensor (6, H, W) is converted to intensity in
    [0, 1] via ``intensity = clamp(1 - t, 0, 1)`` (non-spiking → 0). The
    target is the raw CIFAR-10 image in [0, 1].
    """

    def __init__(self, whitened: Cifar10WhitenedDataset, raw_rgb: torch.Tensor):
        if len(whitened) != raw_rgb.shape[0]:
            raise ValueError("whitened/raw length mismatch")
        self._whitened = whitened
        self._raw = raw_rgb

    def __len__(self) -> int:
        return len(self._whitened)

    def __getitem__(self, idx: int):
        times = self._whitened.all_times[idx]
        finite = torch.isfinite(times)
        intensity = torch.zeros_like(times)
        intensity[finite] = (1.0 - times[finite]).clamp_(0.0, 1.0)
        return intensity, self._raw[idx]


def _raw_cifar10_tensor(data_root: str, train: bool) -> torch.Tensor:
    """(N, 3, 32, 32) float32 CIFAR-10 in [0, 1]."""
    ds = torchvision.datasets.CIFAR10(root=data_root, train=train, download=True)
    images = torch.from_numpy(ds.data).float() / 255.0
    return images.permute(0, 3, 1, 2).contiguous()


class _RgbAutoencoderDataset(Dataset):
    """Yields (rgb_image, rgb_image) pairs for the standard autoencoder mode."""

    def __init__(self, raw_rgb: torch.Tensor):
        self._raw = raw_rgb

    def __len__(self) -> int:
        return self._raw.shape[0]

    def __getitem__(self, idx: int):
        img = self._raw[idx]
        return img, img


def _make_loaders(config: Config) -> tuple[DataLoader, DataLoader]:
    """Build (train, test) loaders yielding (input, target) pairs.

    Bypasses the project's DoG pipeline; loads raw CIFAR-10 directly. For
    ``input_mode == "cifar10_whitened"``, also loads the cached whitened
    spike-time tensors and pairs them with the raw RGB targets.
    """
    train_raw = _raw_cifar10_tensor(config.data_root, train=True)
    test_raw = _raw_cifar10_tensor(config.data_root, train=False)

    if config.input_mode == "cifar10":
        train_ds: Dataset = _RgbAutoencoderDataset(train_raw)
        test_ds: Dataset = _RgbAutoencoderDataset(test_raw)
    elif config.input_mode == "cifar10_whitened":
        train_w = Cifar10WhitenedDataset(config.data_root, "train")
        test_w = Cifar10WhitenedDataset(
            config.data_root, "test", kernels=train_w.kernels, mean=train_w.mean
        )
        train_ds = _WhitenedToRgbDataset(train_w, train_raw)
        test_ds = _WhitenedToRgbDataset(test_w, test_raw)
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


def _rmse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(nn.functional.mse_loss(x, y) + 1e-12)


def _evaluate(
    model: ConvAutoencoder, loader: DataLoader, device: str
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Return mean RMSE on the loader plus a sample (target, recon) for plotting."""
    model.eval()
    total = 0.0
    n_imgs = 0
    sample_target: torch.Tensor | None = None
    sample_recon: torch.Tensor | None = None
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            recon = model(inputs)
            total += _rmse(recon, targets).item() * targets.size(0)
            n_imgs += targets.size(0)
            if sample_target is None:
                sample_target = targets[:8].detach().cpu()
                sample_recon = recon[:8].detach().cpu()
    return total / max(n_imgs, 1), sample_target, sample_recon


def _train_epoch(
    model: ConvAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_bimodal: float,
    device: str,
) -> tuple[float, float, float]:
    model.train()
    n_imgs = 0
    total_recon = 0.0
    total_pen = 0.0
    total_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        recon = model(inputs)
        recon_loss = _rmse(recon, targets)
        pen = bimodal_penalty(model.encoder.weight)
        loss = recon_loss + lambda_bimodal * pen

        loss.backward()
        optimizer.step()
        model.clip_encoder_weights()

        bs = targets.size(0)
        n_imgs += bs
        total_recon += recon_loss.item() * bs
        total_pen += pen.item() * bs
        total_loss += loss.item() * bs

    return total_recon / n_imgs, total_pen / n_imgs, total_loss / n_imgs


_INPUT_CHANNELS = {"cifar10": 3, "cifar10_whitened": 6}


def _run_one(config: Config) -> dict:
    """Train a single configuration, write artifacts, return summary."""
    set_seed(config.seed)
    out_dir = _resolve_output_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=4)

    train_loader, test_loader = _make_loaders(config)
    device = torch.device(config.device)

    in_channels = _INPUT_CHANNELS[config.input_mode]
    model = ConvAutoencoder(
        in_channels=in_channels,
        out_channels=3,  # always reconstruct raw RGB
        num_filters=config.num_filters,
        kernel_size=config.kernel_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    save_weight_histogram(
        model.encoder.weight,
        out_dir / "weight_hist_epoch0.png",
        title=f"Encoder weights — init (λ={config.lambda_bimodal})",
    )

    history = {
        "recon_loss": [],
        "bimodal_loss": [],
        "total_loss": [],
        "val_rmse": [],
        "bimodality_score": [],
    }
    mid_epoch = max(1, config.epochs // 2)

    for epoch in range(1, config.epochs + 1):
        recon, pen, total = _train_epoch(
            model, train_loader, optimizer, config.lambda_bimodal, device
        )
        val_rmse, _, _ = _evaluate(model, test_loader, device)
        score = bimodality_score(model.encoder.weight)

        history["recon_loss"].append(recon)
        history["bimodal_loss"].append(pen)
        history["total_loss"].append(total)
        history["val_rmse"].append(val_rmse)
        history["bimodality_score"].append(score)

        logger.info(
            "epoch %3d/%d | recon %.4f | pen %.4f | total %.4f | val RMSE %.4f | bimod %.3f",
            epoch,
            config.epochs,
            recon,
            pen,
            total,
            val_rmse,
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
            model.encoder.weight,
            out_dir / "filter_grid_final.png",
        )
    else:
        save_filter_grid(
            model.encoder.weight,
            out_dir / "filter_grid_final.png",
            rescale_per_filter=True,
        )
        save_filter_grid(
            model.encoder.weight,
            out_dir / "filter_grid_final_raw.png",
            rescale_per_filter=False,
        )
    save_loss_curves(history, out_dir / "loss_curves.png")
    save_bimodality_curve(history, out_dir / "bimodality_curve.png")

    final_val_rmse, sample_target, sample_recon = _evaluate(model, test_loader, device)
    save_reconstructions(
        sample_target, sample_recon, out_dir / "reconstructions.png", n=8
    )

    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "decoder": model.decoder.state_dict(),
        },
        out_dir / "checkpoint.pt",
    )

    summary = {
        "config": asdict(config),
        "final_recon_rmse": float(final_val_rmse),
        "final_bimodality": float(history["bimodality_score"][-1]),
        "final_bimodal_penalty": float(history["bimodal_loss"][-1]),
        "history": history,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=4)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=4)

    logger.info(
        "done — val RMSE %.4f | bimodality %.3f → %s",
        final_val_rmse,
        history["bimodality_score"][-1],
        out_dir,
    )
    return summary


def _parse_lambda_sweep(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument(
        "--input-mode",
        default="cifar10",
        choices=list(INPUT_MODES),
        help="cifar10: 3ch raw RGB → RGB. cifar10_whitened: 6ch whitened spike intensities → RGB.",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--num-filters", type=int, default=256)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-bimodal", type=float, default=0.1)
    parser.add_argument(
        "--lambda-bimodal-sweep",
        default="",
        help='Comma-separated λ values, e.g. "0.0,0.01,0.1,1.0". Overrides --lambda-bimodal.',
    )
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    raw = vars(args)
    sweep_raw = raw.pop("lambda_bimodal_sweep", "")
    base_cfg_kwargs = {k: v for k, v in raw.items() if k in Config.__dataclass_fields__}

    if sweep_raw:
        lambdas = _parse_lambda_sweep(sweep_raw)
        records = []
        sweep_root = (
            Path(args.output_dir)
            if args.output_dir
            else Path(
                f"logs/ann_bimodal_autoencoder/{args.input_mode}/seed_{args.seed}/sweep"
            )
        )
        sweep_root.mkdir(parents=True, exist_ok=True)

        for lam in lambdas:
            cfg_kwargs = dict(base_cfg_kwargs)
            cfg_kwargs["lambda_bimodal"] = lam
            cfg_kwargs["output_dir"] = str(sweep_root / f"lambda_{lam:g}")
            cfg = Config(**cfg_kwargs)
            logger.info("=== sweep λ=%g ===", lam)
            summary = _run_one(cfg)
            records.append(
                {
                    "lambda_bimodal": lam,
                    "final_recon_rmse": summary["final_recon_rmse"],
                    "final_bimodality": summary["final_bimodality"],
                    "final_bimodal_penalty": summary["final_bimodal_penalty"],
                }
            )

        with open(sweep_root / "sweep_records.json", "w") as f:
            json.dump(records, f, indent=4)
        save_lambda_sweep(records, sweep_root / "lambda_sweep.png")
        logger.info("sweep summary written to %s", sweep_root)
    else:
        cfg = Config(**base_cfg_kwargs)
        _run_one(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
