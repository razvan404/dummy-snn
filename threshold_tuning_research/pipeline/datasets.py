from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DatasetSpec:
    """One registered dataset. Adding a dataset = one `register(...)` call, instead of
    editing argparse `choices` lists and `if dataset == ...` branches across the pipeline."""

    name: str  # CLI name
    hyperparams_key: str  # key into threshold_tuning_research.paper_hyperparams._CONFIGS
    load_split: Callable[[int | None], tuple[dict, dict]]  # -> (train, test), each {images, labels}
    load_train_images: Callable[[int | None], torch.Tensor]


_REGISTRY: dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec) -> None:
    _REGISTRY[spec.name] = spec


def dataset_names() -> list[str]:
    return list(_REGISTRY)


def get(dataset: str) -> DatasetSpec:
    if dataset not in _REGISTRY:
        raise ValueError(f"Unknown dataset {dataset!r}. Registered: {dataset_names()}")
    return _REGISTRY[dataset]


def _resolve_num_bins(dataset: str, num_bins: int | None) -> int | None:
    if num_bins is not None:
        return num_bins
    from threshold_tuning_research.paper_hyperparams import get_paper_hyperparams

    return get_paper_hyperparams(get(dataset).hyperparams_key).get("num_bins", 64)


def load_split_data(dataset: str, num_bins: int | None = None) -> tuple[dict, dict]:
    return get(dataset).load_split(_resolve_num_bins(dataset, num_bins))


def load_train_images(dataset: str, num_bins: int | None = None) -> torch.Tensor:
    return get(dataset).load_train_images(_resolve_num_bins(dataset, num_bins))


# --- CIFAR-10 (ZCA whitened) ---
def _cifar10_split(num_bins):
    from threshold_tuning_research.datasets import Cifar10WhitenedDataset

    train_ds = Cifar10WhitenedDataset("data", "train", num_bins=num_bins)
    test_ds = Cifar10WhitenedDataset(
        "data", "test", num_bins=num_bins, kernels=train_ds.kernels, mean=train_ds.mean
    )
    return (
        {"images": train_ds.all_times, "labels": train_ds.outputs},
        {"images": test_ds.all_times, "labels": test_ds.outputs},
    )


def _cifar10_train_images(num_bins):
    from threshold_tuning_research.datasets import Cifar10WhitenedDataset

    return Cifar10WhitenedDataset("data", "train", num_bins=num_bins).all_times


# --- Fashion-MNIST (DoG) ---
def _fashion_split(num_bins):
    from threshold_tuning_research.datasets import FashionMnistDataset

    train_ds = FashionMnistDataset(
        "data", "train", cache_path="data/fashion_mnist_cache/train_dog.pt", num_bins=num_bins
    )
    test_ds = FashionMnistDataset(
        "data", "test", cache_path="data/fashion_mnist_cache/test_dog.pt", num_bins=num_bins
    )
    return (
        {"images": train_ds.all_times, "labels": train_ds.outputs},
        {"images": test_ds.all_times, "labels": test_ds.outputs},
    )


def _fashion_train_images(num_bins):
    from threshold_tuning_research.datasets import FashionMnistDataset

    return FashionMnistDataset(
        "data", "train", cache_path="data/fashion_mnist_cache/train_dog.pt", num_bins=num_bins
    ).all_times


# --- generic preprocessed (e.g. mnist) ---
def _processed_split(dataset):
    d = f"data/processed-{dataset}"
    return (
        torch.load(f"{d}/train.pt", weights_only=True),
        torch.load(f"{d}/test.pt", weights_only=True),
    )


def _processed_train_images(dataset):
    return torch.load(f"data/processed-{dataset}/train.pt", weights_only=True)["images"]


register(DatasetSpec("cifar10", "cifar10", _cifar10_split, _cifar10_train_images))
register(DatasetSpec("fashion_mnist", "fashion_mnist", _fashion_split, _fashion_train_images))
register(
    DatasetSpec(
        "mnist",
        "mnist",
        lambda nb: _processed_split("mnist"),
        lambda nb: _processed_train_images("mnist"),
    )
)
