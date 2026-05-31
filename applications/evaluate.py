import argparse
import gc
import json
import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from applications.common import load_split_data, resolve_params
from applications.pipeline.datasets import dataset_names
from spiking.evaluation import evaluate_classifier
from spiking.evaluation.conv_feature_extraction import sum_pool_features
from spiking.evaluation.feature_extraction import spike_times_to_features
from spiking.utils.checkpoints import load_model

logger = logging.getLogger(__name__)


def output_filters(model) -> int:
    """Number of filters of the model's output layer — works for a single conv-IF layer
    or a SpikingSequential stack (the last layer carrying a num_filters)."""
    from spiking.layers import SpikingSequential

    if isinstance(model, SpikingSequential):
        for layer in reversed(model.layers):
            if getattr(layer, "num_filters", None):
                return layer.num_filters
        raise ValueError("no layer with num_filters in the stack")
    return model.num_filters


@torch.no_grad()
def _extract_features(
    layer,
    images: torch.Tensor,
    labels: torch.Tensor,
    pool_size: int,
    t_target: float | None,
    chunk_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    layer.eval()
    layer.to(device)
    flat_dim = output_filters(layer) * pool_size * pool_size
    X = np.empty((len(images), flat_dim), dtype=np.float32)

    for start in tqdm(
        range(0, len(images), chunk_size),
        desc="Extracting features",
        unit="batch",
        leave=False,
    ):
        end = min(start + chunk_size, len(images))
        st = layer.infer_spike_times_batch(images[start:end].to(device))
        feat = spike_times_to_features(st.cpu(), t_target=t_target)
        pooled = sum_pool_features(feat, pool_size)
        X[start:end] = pooled.flatten(1).numpy()
        del st, feat, pooled

    layer.cpu()
    return X, labels.numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate conv SNN with a Torch LinearSVC"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=dataset_names(),
    )
    parser.add_argument("--num-filters", type=int, default=None)
    parser.add_argument("--t-obj", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--layer2-filters", type=int, default=None,
        help="If set, evaluate the 2-layer stack whose L1 is (--num-filters, --t-obj).",
    )
    parser.add_argument("--layer2-tobj", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    nf, t_obj, model_dir = resolve_params(args)
    if args.layer2_filters is not None:
        from applications.pipeline import LayerSpec, RunSpec

        spec = RunSpec(
            args.dataset,
            args.seed,
            (LayerSpec(nf, t_obj), LayerSpec(args.layer2_filters, args.layer2_tobj)),
        )
        model_dir = str(spec.model_dir)

    if not os.path.exists(f"{model_dir}/model.pth"):
        logger.error("No model at %s — run train.py first", model_dir)
        return
    if not args.force and os.path.exists(f"{model_dir}/metrics.json"):
        logger.info("Metrics exist at %s (use --force)", model_dir)
        return

    with open(f"{model_dir}/setup.json") as f:
        setup = json.load(f)
    pool_size = setup.get("pool_size", 2)
    t_target = setup.get("target_timestamp")

    layer = load_model(f"{model_dir}/model.pth")
    train_data, test_data = load_split_data(args.dataset, setup.get("num_bins"))

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

    # LinearSVC (Torch)
    svc_train, svc_val = evaluate_classifier(X_train, y_train, X_test, y_test)
    logger.info(
        "LinearSVC  — train: %.4f, val: %.4f",
        svc_train["accuracy"],
        svc_val["accuracy"],
    )

    metrics = {
        "linear_svc": {"train": svc_train, "validation": svc_val},
    }
    with open(f"{model_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Saved metrics to %s/metrics.json", model_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
