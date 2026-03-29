import argparse

from applications.datasets import DATASETS

from .config import NUM_EPOCHS, SEEDS
from .step1_train import run as run_step1
from .step2_optimize import run as run_step2
from .step3_predict import run as run_step3


def main():
    parser = argparse.ArgumentParser(
        description="Threshold Prediction Experiment (all steps)"
    )
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3], help="Run only this step (default: all)"
    )
    args = parser.parse_args()

    seeds = args.seeds
    steps = [args.step] if args.step else [1, 2, 3]

    if 1 in steps:
        run_step1(args.dataset, num_epochs=args.epochs, force=args.force, seeds=seeds)
    if 2 in steps:
        run_step2(args.dataset, force=args.force, seeds=seeds)
    if 3 in steps:
        run_step3(args.dataset, seeds=seeds)


if __name__ == "__main__":
    main()
