import argparse
import json
import logging

import torch

from threshold_tuning_research.common import set_seed
from threshold_tuning_research.paper_hyperparams import get_paper_hyperparams
from threshold_tuning_research.pipeline import LayerSpec, RunSpec
from threshold_tuning_research.pipeline.datasets import dataset_names, load_train_images
from threshold_tuning_research.pipeline.featurize import featurize_through
from threshold_tuning_research.train import _build_and_train_layer
from spikinn.layers import SpikeTimeMinPool, SpikinnSequential
from spikinn.utils.checkpoints import load_model, save_model

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train layer 2 on a frozen, featurized layer-1 prefix (2-layer stack)"
    )
    parser.add_argument("--dataset", default="cifar10", choices=dataset_names())
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--l1-filters", type=int, required=True)
    parser.add_argument("--l1-tobj", type=float, required=True)
    parser.add_argument("--l2-filters", type=int, required=True)
    parser.add_argument("--l2-tobj", type=float, required=True, help="must exceed --l1-tobj")
    parser.add_argument("--pool", type=int, default=2, help="min-pool kernel/stride between L1 and L2")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    set_seed(args.seed)
    # RunSpec validates t_obj(L2) > t_obj(L1).
    spec = RunSpec(
        args.dataset,
        args.seed,
        (LayerSpec(args.l1_filters, args.l1_tobj), LayerSpec(args.l2_filters, args.l2_tobj)),
    )
    l1_dir = spec.prefix_dir
    logger.info("Loading frozen layer-1 from %s", l1_dir)
    l1 = load_model(f"{l1_dir}/model.pth")
    minpool = SpikeTimeMinPool(args.pool)

    logger.info("Featurizing train set through frozen (L1 + min-pool)...")
    images = load_train_images(args.dataset)
    feat_device = "cuda" if torch.cuda.is_available() else "cpu"
    maps = featurize_through(
        SpikinnSequential(l1, minpool), images, device=feat_device, chunk_size=args.chunk_size
    )
    logger.info("  layer-1 featurized maps: %s", tuple(maps.shape))

    params = get_paper_hyperparams(args.dataset)
    params["num_filters"] = args.l2_filters
    params["target_timestamp"] = args.l2_tobj
    if args.num_epochs is not None:
        params["num_epochs"] = args.num_epochs

    layer2, training_logs = _build_and_train_layer(maps, params, device=args.device)
    model = SpikinnSequential(l1, minpool, layer2)

    out_dir = spec.model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, f"{out_dir}/model.pth")
    torch.save(training_logs, f"{out_dir}/training_logs.pt")
    setup = {
        "dataset": args.dataset,
        "seed": args.seed,
        "layers": [
            {"num_filters": args.l1_filters, "t_obj": args.l1_tobj},
            {"num_filters": args.l2_filters, "t_obj": args.l2_tobj},
        ],
        "pool": args.pool,
        "pool_size": params["pool_size"],
        "num_bins": params["num_bins"],
        "target_timestamp": args.l2_tobj,  # last layer drives the decoder
    }
    with open(f"{out_dir}/setup.json", "w") as f:
        json.dump(setup, f, indent=4)
    logger.info("Saved 2-layer model to %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
