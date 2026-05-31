import pytest
import torch

from spiking.layers import SpikeTimeMinPool, SpikingSequential
from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer
from spiking.threshold.constant_initialization import ConstantInitialization

INF = float("inf")


class TestSpikeTimeMinPool:
    def test_earliest_spike_wins(self):
        x = torch.tensor([[[[0.2, 0.5], [0.9, 0.1]]]])
        out = SpikeTimeMinPool(2).infer_spike_times_batch(x)
        assert out.shape == (1, 1, 1, 1)
        assert out.item() == pytest.approx(0.1)  # earliest spike in the window

    def test_all_nonfiring_window_stays_inf(self):
        x = torch.full((1, 1, 2, 2), INF)
        assert SpikeTimeMinPool(2).infer_spike_times_batch(x).item() == INF

    def test_partial_window_picks_finite_min(self):
        x = torch.tensor([[[[INF, 0.7], [INF, INF]]]])
        assert SpikeTimeMinPool(2).infer_spike_times_batch(x).item() == pytest.approx(0.7)

    def test_channels_preserved_and_shape(self):
        x = torch.rand(4, 8, 12, 12)
        out = SpikeTimeMinPool(kernel_size=2, stride=2).infer_spike_times_batch(x)
        assert out.shape == (4, 8, 6, 6)

    def test_stride_independent_of_kernel(self):
        x = torch.rand(1, 1, 8, 8)
        out = SpikeTimeMinPool(kernel_size=3, stride=2).infer_spike_times_batch(x)
        assert out.shape == (1, 1, 3, 3)  # (8-3)//2 + 1

    def test_unbatched_inference(self):
        x = torch.rand(2, 4, 4)
        out = SpikeTimeMinPool(2).infer_spike_times(x)
        assert out.shape == (2, 2, 2)

    def test_composes_in_two_layer_stack(self):
        init = ConstantInitialization(1.0)
        c1 = ConvIntegrateAndFireLayer(2, 8, 3, 1, 0, init, refractory_period=INF)
        c2 = ConvIntegrateAndFireLayer(8, 6, 3, 1, 0, init, refractory_period=INF)
        torch.nn.init.uniform_(c1.weights, 0, 1)
        torch.nn.init.uniform_(c2.weights, 0, 1)
        model = SpikingSequential(c1, SpikeTimeMinPool(2, 2), c2)
        inp = torch.rand(4, 2, 12, 12)
        inp[inp > 0.7] = INF
        out = model.infer_spike_times_batch(inp)
        # 12 -conv3-> 10 -pool2-> 5 -conv3-> 3
        assert out.shape == (4, 6, 3, 3)
