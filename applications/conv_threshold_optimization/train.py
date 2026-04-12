"""Train a convolutional integrate-and-fire SNN with STDP."""

import argparse
import logging
import os

from applications.common import resolve_params
from applications.conv_learning.train import train_model

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a conv SNN model")
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"],
    )
    parser.add_argument("--num-filters", type=int, default=None)
    parser.add_argument("--t-obj", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    nf, t_obj, model_dir = resolve_params(args)

    if not args.force and os.path.exists(f"{model_dir}/model.pth"):
        logger.info("Model exists at %s (use --force to retrain)", model_dir)
        return

    train_model(
        dataset=args.dataset,
        seed=args.seed,
        t_obj=t_obj,
        num_filters=nf,
        output_dir=model_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
