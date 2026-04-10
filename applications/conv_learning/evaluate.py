import argparse
import gc
import json
import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from spiking.evaluation import evaluate_classifier
from spiking.evaluation.conv_feature_extraction import sum_pool_features
from spiking.evaluation.feature_extraction import spike_times_to_features
from spiking.utils.checkpoints import load_model

logger = logging.getLogger(__name__)


@torch.no_grad()
def _evaluate_split(
    layer,
    images: torch.Tensor,
    labels: torch.Tensor,
    pool_size: int = 2,
    t_target: float | None = None,
    chunk_size: int = 512,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract conv features from full images with sum pooling.

    :param layer: Trained ConvIntegrateAndFireLayer.
    :param images: (N, C, H, W) spike-encoded images.
    :param labels: (N,) integer labels.
    :param pool_size: Sum pooling grid size.
    :param t_target: Target time for feature conversion (Falez Eq 10).
    :param chunk_size: Batch size for inference.
    :param device: Device for inference ('cpu' or 'cuda').
    :returns: (X, y) numpy arrays.
    """
    layer.eval()
    layer.to(device)
    flat_dim = layer.num_filters * pool_size * pool_size
    X = np.empty((len(images), flat_dim), dtype=np.float32)

    for start in tqdm(
        range(0, len(images), chunk_size),
        desc="Extracting features",
        unit="batch",
        leave=False,
    ):
        end = min(start + chunk_size, len(images))
        chunk = images[start:end].to(device)
        spike_times = layer.infer_spike_times_batch(chunk)
        features = spike_times_to_features(spike_times.cpu(), t_target=t_target)
        pooled = sum_pool_features(features, pool_size)
        X[start:end] = pooled.flatten(1).numpy()
        del spike_times, features, pooled, chunk

    return X, labels.numpy()


def _load_data(dataset: str, processed_dir: str | None) -> tuple[dict, dict]:
    """Load train and test data, using dataset classes when possible."""
    if dataset == "cifar10" and processed_dir is None:
        from applications.datasets import Cifar10WhitenedDataset

        logger.info("Loading CIFAR-10 via Cifar10WhitenedDataset (rho=1.0)...")
        train_ds = Cifar10WhitenedDataset("data", "train")
        test_ds = Cifar10WhitenedDataset(
            "data",
            "test",
            kernels=train_ds.kernels,
            mean=train_ds.mean,
        )
        train_data = {"images": train_ds.all_times, "labels": train_ds.outputs}
        test_data = {"images": test_ds.all_times, "labels": test_ds.outputs}
    else:
        if processed_dir is None:
            processed_dir = f"data/processed-{dataset}"
        logger.info("Loading preprocessed data from %s...", processed_dir)
        train_data = torch.load(f"{processed_dir}/train.pt", weights_only=True)
        test_data = torch.load(f"{processed_dir}/test.pt", weights_only=True)
    return train_data, test_data


def evaluate_model_dir(
    model_dir: str,
    device: str = "cpu",
    chunk_size: int = 512,
):
    """Evaluate a trained model from its output directory.

    Reads model.pth and setup.json, loads the matching preprocessed dataset,
    extracts features, fits Ridge classifier, saves metrics.json.

    :param model_dir: Directory containing model.pth and setup.json.
    :param device: Device for inference ('cpu' or 'cuda').
    :param chunk_size: Batch size for conv inference.
    :returns: Dict with 'train' and 'validation' metric dicts.
    """
    # Load setup
    with open(f"{model_dir}/setup.json") as f:
        setup = json.load(f)

    dataset = setup["dataset"]
    processed_dir = setup.get("processed_dir")
    pool_size = setup.get("pool_size", 2)
    t_target = setup.get("target_timestamp")

    logger.info("Evaluating %s model from %s (device=%s)", dataset, model_dir, device)

    # Load model
    layer = load_model(f"{model_dir}/model.pth")

    # Load preprocessed data
    train_data, test_data = _load_data(dataset, processed_dir)

    # Extract features
    logger.info("Extracting train features (%d images)...", len(train_data["images"]))
    X_train, y_train = _evaluate_split(
        layer,
        train_data["images"],
        train_data["labels"],
        pool_size=pool_size,
        t_target=t_target,
        chunk_size=chunk_size,
        device=device,
    )
    del train_data
    gc.collect()

    logger.info("Extracting test features (%d images)...", len(test_data["images"]))
    X_test, y_test = _evaluate_split(
        layer,
        test_data["images"],
        test_data["labels"],
        pool_size=pool_size,
        t_target=t_target,
        chunk_size=chunk_size,
        device=device,
    )
    del test_data
    gc.collect()

    # Classify
    train_m, val_m = evaluate_classifier(X_train, y_train, X_test, y_test)
    logger.info("  Train accuracy: %.4f", train_m["accuracy"])
    logger.info("  Test accuracy:  %.4f", val_m["accuracy"])

    # Save metrics
    metrics = {"train": train_m, "validation": val_m}
    with open(f"{model_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Saved metrics to %s/metrics.json", model_dir)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Evaluate trained conv SNN models")
    parser.add_argument(
        "model_dirs",
        type=str,
        nargs="+",
        help="Directories containing model.pth and setup.json "
        "(supports globs, e.g. data/cifar10_whitened_patches/tobj_0.97/seed_*)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (cpu or cuda)",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even if metrics.json already exists",
    )
    args = parser.parse_args()

    for model_dir in args.model_dirs:
        if not os.path.exists(f"{model_dir}/model.pth"):
            logger.warning("No model.pth in %s, skipping", model_dir)
            continue
        if not args.force and os.path.exists(f"{model_dir}/metrics.json"):
            logger.info(
                "Skipping %s (already evaluated, use --force to re-evaluate)", model_dir
            )
            continue
        evaluate_model_dir(
            model_dir=model_dir,
            device=args.device,
            chunk_size=args.chunk_size,
        )
