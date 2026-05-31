import torch

from threshold_tuning_research.common import set_seed
from threshold_tuning_research.evaluate import output_filters
from threshold_tuning_research.paper_hyperparams import get_paper_hyperparams
from threshold_tuning_research.pipeline.featurize import featurize_through
from threshold_tuning_research.train import _build_and_train_layer
from spikinn.layers import SpikeTimeMinPool, SpikinnSequential


def _tiny_images(n=8, channels=6, hw=18, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, channels, hw, hw, generator=g)
    x[x > 0.8] = float("inf")  # non-firing inputs
    return x


def _l2_params(num_filters, t_obj):
    p = get_paper_hyperparams("cifar10")
    p.update(num_filters=num_filters, target_timestamp=t_obj, num_epochs=1)
    return p


def test_two_layer_train_and_infer_end_to_end():
    # Mirrors threshold_tuning_research/train_multilayer.py with tiny layers: build+train L1, featurize
    # through (L1 + min-pool), train L2 on the maps, assemble the stack, infer.
    set_seed(0)
    images = _tiny_images()

    l1, _ = _build_and_train_layer(images, _l2_params(num_filters=4, t_obj=0.50))
    minpool = SpikeTimeMinPool(2)

    maps = featurize_through(SpikinnSequential(l1, minpool), images, chunk_size=4)
    # 18 -conv5-> 14 -pool2-> 7 ; 4 filters
    assert maps.shape == (8, 4, 7, 7)

    l2, _ = _build_and_train_layer(maps, _l2_params(num_filters=3, t_obj=0.70))
    model = SpikinnSequential(l1, minpool, l2)

    out = model.infer_spike_times_batch(images)
    # 7 -conv5-> 3 ; 3 filters
    assert out.shape == (8, 3, 3, 3)
    assert output_filters(model) == 3  # depth-agnostic feature dim for evaluate


def test_output_filters_single_vs_stack():
    from spikinn.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer
    from spikinn.threshold.constant_initialization import ConstantInitialization

    init = ConstantInitialization(1.0)
    c1 = ConvIntegrateAndFireLayer(2, 5, 5, 1, 0, init, refractory_period=float("inf"))
    c2 = ConvIntegrateAndFireLayer(5, 7, 5, 1, 0, init, refractory_period=float("inf"))
    assert output_filters(c1) == 5  # single layer
    assert output_filters(SpikinnSequential(c1, SpikeTimeMinPool(2), c2)) == 7  # stack
