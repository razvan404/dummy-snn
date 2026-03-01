import sys

from applications.datasets import create_dataset

DATASETS = ["mnist", "fashion_mnist", "cifar10", "mnist_subset"]


def main():
    failed = []
    for name in DATASETS:
        try:
            train_loader, test_loader = create_dataset(name)
            train_size = len(train_loader.dataset)
            test_size = len(test_loader.dataset)
            print(f"  {name}: OK (train={train_size}, test={test_size})")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            failed.append(name)

    if failed:
        print(f"\nFailed datasets: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\nAll datasets OK.")


if __name__ == "__main__":
    main()
