from torch.utils.data import DataLoader

from .base import SpikeEncodingDataset
from .cifar10 import Cifar10Dataset
from .fashion_mnist import FashionMnistDataset
from .fer2013 import Fer2013Dataset
from .mnist import MnistDataset
from .mnist_subset import MnistSubsetDataset
from .cifar10_whitened import Cifar10WhitenedDataset, create_cifar10_whitened
from .cifar10_patches import Cifar10PatchDataset


def create_dataset(name: str, root_dir: str = "data") -> tuple[DataLoader, DataLoader]:
    """Load train and test datasets by name. Returns (train_loader, test_loader).

    Access the dataset's native image shape via train_loader.dataset.image_shape.
    """
    if name == "mnist_subset":
        train_dataset = MnistSubsetDataset(f"{root_dir}/mnist-subset", "train", image_shape=(16, 16))
        test_dataset = MnistSubsetDataset(f"{root_dir}/mnist-subset", "test", image_shape=(16, 16))
    elif name == "mnist":
        train_dataset = MnistDataset(root_dir, "train")
        test_dataset = MnistDataset(root_dir, "test")
    elif name == "fashion_mnist":
        train_dataset = FashionMnistDataset(root_dir, "train")
        test_dataset = FashionMnistDataset(root_dir, "test")
    elif name == "cifar10":
        train_dataset = Cifar10Dataset(root_dir, "train")
        test_dataset = Cifar10Dataset(root_dir, "test")
    elif name == "fer2013":
        train_dataset = Fer2013Dataset("train")
        test_dataset = Fer2013Dataset("test")
    elif name == "cifar10_whitened":
        return create_cifar10_whitened()
    else:
        raise ValueError(f"unknown dataset: {name!r}")

    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False)
    return train_loader, test_loader


DATASETS = ["mnist", "mnist_subset", "fashion_mnist", "cifar10", "cifar10_whitened", "fer2013"]
