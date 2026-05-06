"""One-off eval for a SNN model living at an arbitrary directory.

Derives ``t_target`` for spike-time-to-feature decoding from the training set's
spike-time statistics (median of finite spike times across a sample), then uses
the same value for both train and validation feature extraction. This gives a
training-calibrated, target-timestamp-agnostic eval — important for weight_mean
runs where ``setup.json['target_timestamp']`` is meaningless.

Usage: python -m applications._eval_wmean <model_dir> --dataset fashion_mnist
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import torch

from applications.common import load_split_data
from applications.conv_threshold_optimization.evaluate import _extract_features
from spiking.evaluation import evaluate_classifier
from spiking.evaluation.ridge_column_swap import RidgeColumnSwap
from spiking.utils.checkpoints import load_model

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_t_target(
    layer,
    images: torch.Tensor,
    device: str,
    n_sample: int = 4096,
    quantile: float = 0.5,
) -> float:
    """Estimate t_target as a quantile of finite spike times on a training sample.

    Robust to non-spiking neurons (filtered out via isfinite) and to outliers
    (default median). Calibrates feature decoding to the network's actual
    operating regime regardless of the training-time target_timestamp.
    """
    sample = images[: min(n_sample, len(images))].to(device)
    layer.eval()
    layer.to(device)
    st = layer.infer_spike_times_batch(sample)
    finite = st[torch.isfinite(st)]
    if finite.numel() == 0:
        return 1.0
    # torch.quantile caps at ~16M elements; subsample if needed.
    if finite.numel() > 1_000_000:
        idx = torch.randint(finite.numel(), (1_000_000,), device=finite.device)
        finite = finite[idx]
    t = finite.quantile(quantile).item()
    layer.cpu()
    return float(t)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--dataset", choices=["mnist", "cifar10", "fashion_mnist"], required=True
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument(
        "--t-target",
        type=float,
        default=None,
        help="Override training-derived t_target; default = training median spike time.",
    )
    parser.add_argument("--n-sample", type=int, default=4096)
    parser.add_argument("--quantile", type=float, default=0.5)
    args = parser.parse_args()

    setup = json.loads((args.model_dir / "setup.json").read_text())
    pool_size = setup.get("pool_size", 2)

    layer = load_model(str(args.model_dir / "model.pth"))
    train_data, test_data = load_split_data(args.dataset)

    if args.t_target is not None:
        t_target = args.t_target
        logger.info("Using user-specified t_target=%.4f", t_target)
    else:
        t_target = compute_t_target(
            layer, train_data["images"], args.device, args.n_sample, args.quantile
        )
        logger.info(
            "Derived training t_target (q=%.2f, n=%d) = %.4f",
            args.quantile,
            args.n_sample,
            t_target,
        )

    logger.info("Extracting train features (%d images)...", len(train_data["images"]))
    X_train, y_train = _extract_features(
        layer,
        train_data["images"],
        train_data["labels"],
        pool_size,
        t_target,
        args.chunk_size,
        args.device,
    )
    del train_data
    gc.collect()

    logger.info("Extracting test features (%d images)...", len(test_data["images"]))
    X_test, y_test = _extract_features(
        layer,
        test_data["images"],
        test_data["labels"],
        pool_size,
        t_target,
        args.chunk_size,
        args.device,
    )
    del test_data
    gc.collect()

    svc_train, svc_val = evaluate_classifier(X_train, y_train, X_test, y_test)
    logger.info(
        "LinearSVC — train %.4f val %.4f", svc_train["accuracy"], svc_val["accuracy"]
    )

    ridge = RidgeColumnSwap(alpha=1.0)
    ridge_train, ridge_val = evaluate_classifier(
        X_train, y_train, X_test, y_test, classifier=ridge
    )
    logger.info(
        "Ridge     — train %.4f val %.4f",
        ridge_train["accuracy"],
        ridge_val["accuracy"],
    )

    metrics = {
        "linear_svc": {"train": svc_train, "validation": svc_val},
        "ridge": {"train": ridge_train, "validation": ridge_val},
        "eval_t_target": t_target,
        "eval_t_target_source": "user"
        if args.t_target is not None
        else "train_quantile",
    }
    out = args.model_dir / "metrics.json"
    out.write_text(json.dumps(metrics, indent=4))
    logger.info("Saved %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
