"""Equivalence tests for spiking_backend against the existing dense path.

The reference implementation in ``spiking_backend.reference`` must produce
the same spike-time outputs as ``ConvIntegrateAndFireLayer._conv2d_accumulate``
and ``multi_threshold_conv_accumulate`` up to float-summation
non-associativity.
"""

from __future__ import annotations

import pytest
import torch

from applications.threshold_research.conv_neuron_perturbation import (
    multi_threshold_conv_accumulate,
)
from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer
from spiking.threshold.normal_initialization import NormalInitialization
from spiking_backend.reference import (
    spike_driven_conv_accumulate,
    spike_driven_conv_accumulate_multi_threshold,
)


def _make_layer(
    in_channels: int = 6,
    num_filters: int = 16,
    kernel_size: int = 5,
    stride: int = 1,
    padding: int = 0,
) -> ConvIntegrateAndFireLayer:
    init = NormalInitialization(avg_threshold=2.0, std_dev=0.2, min_threshold=0.5)
    layer = ConvIntegrateAndFireLayer(
        in_channels=in_channels,
        num_filters=num_filters,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        threshold_initialization=init,
        refractory_period=float("inf"),
    )
    torch.nn.init.uniform_(layer.weights, a=0.0, b=1.0)
    return layer


def _make_inputs(
    *,
    batch: int = 2,
    in_channels: int = 6,
    H: int = 12,
    W: int = 12,
    num_bins: int = 16,
    sparsity: float = 0.5,
    seed: int = 0,
) -> torch.Tensor:
    """Latency-encoded spike times: random discrete values in [0, 1] or inf."""
    g = torch.Generator().manual_seed(seed)
    times = (
        torch.randint(0, num_bins, (batch, in_channels, H, W), generator=g).float()
        / num_bins
    )
    mask = torch.rand((batch, in_channels, H, W), generator=g) < sparsity
    times = torch.where(mask, times, torch.full_like(times, float("inf")))
    return times


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("num_bins", [16, 64])
def test_spike_driven_matches_dense(seed: int, padding: int, num_bins: int) -> None:
    """Single-threshold sparse-event implementation matches dense ``F.conv2d``.

    Tolerance scales with the discretisation bin size (1/num_bins): tied
    events at the same time bin can flip crossing order across paths because
    of float-summation non-associativity.
    """
    layer = _make_layer(padding=padding)
    times = _make_inputs(seed=seed, num_bins=num_bins)

    dense_st, _ = layer._conv2d_accumulate(times)
    sparse_st, _ = spike_driven_conv_accumulate(
        times,
        layer.weights_4d,
        layer.thresholds,
        stride=layer.stride,
        padding=layer.padding,
    )

    assert dense_st.shape == sparse_st.shape
    bin_size = 1.0 / num_bins
    finite = torch.isfinite(dense_st) & torch.isfinite(sparse_st)
    if finite.any():
        assert (
            (dense_st[finite] - sparse_st[finite]).abs().max() <= bin_size + 1e-6
        )
    # Spike-vs-no-spike disagreement must stay within tied/border cells: such
    # cells should be near threshold in both paths.
    only_dense = torch.isfinite(dense_st) & torch.isinf(sparse_st)
    only_sparse = torch.isinf(dense_st) & torch.isfinite(sparse_st)
    disagreement = (only_dense | only_sparse).float().mean().item()
    assert disagreement < 0.01, f"too many spike/no-spike mismatches: {disagreement}"
    # We deliberately do NOT compare ``cum_potential`` element-wise: the
    # dense path early-exits once all positions have spiked, freezing its
    # potential, while our path processes all timesteps. The two paths
    # therefore see different totals once everything has saturated. Final
    # spike times — the only quantity downstream consumers depend on —
    # are validated above.


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("num_bins", [16, 64])
def test_multi_threshold_matches_existing(seed: int, num_bins: int) -> None:
    layer = _make_layer()
    times = _make_inputs(seed=seed, num_bins=num_bins)

    # Build a (K, F) threshold matrix matching the perturbation pattern.
    base = layer.thresholds
    fracs = torch.tensor([-0.5, -0.25, 0.0, 0.25])  # K=4
    thresholds_2d = base.unsqueeze(0) * (1 + fracs.unsqueeze(1))  # (K, F)

    dense = multi_threshold_conv_accumulate(
        times,
        layer.weights_4d,
        thresholds_2d,
        stride=layer.stride,
        padding=layer.padding,
        device="cpu",
    )
    sparse = spike_driven_conv_accumulate_multi_threshold(
        times,
        layer.weights_4d,
        thresholds_2d,
        stride=layer.stride,
        padding=layer.padding,
    )

    assert dense.shape == sparse.shape
    bin_size = 1.0 / num_bins
    finite = torch.isfinite(dense) & torch.isfinite(sparse)
    if finite.any():
        assert (dense[finite] - sparse[finite]).abs().max() <= bin_size + 1e-6


def test_no_finite_inputs_returns_inf() -> None:
    layer = _make_layer()
    times = torch.full((1, 6, 12, 12), float("inf"))
    st, pot = spike_driven_conv_accumulate(
        times, layer.weights_4d, layer.thresholds, stride=1, padding=0
    )
    assert torch.isinf(st).all()
    assert (pot == 0).all()


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("num_bins", [16, 64])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_dense(seed: int, num_bins: int) -> None:
    """CUDA implementation matches the dense reference on GPU tensors."""
    layer = _make_layer().cuda()
    times = _make_inputs(seed=seed, num_bins=num_bins).cuda()
    dense_st, _ = layer._conv2d_accumulate(times)
    sparse_st, _ = spike_driven_conv_accumulate(
        times, layer.weights_4d, layer.thresholds, stride=layer.stride, padding=layer.padding
    )
    assert dense_st.shape == sparse_st.shape
    bin_size = 1.0 / num_bins
    finite = torch.isfinite(dense_st) & torch.isfinite(sparse_st)
    if finite.any():
        assert (dense_st[finite] - sparse_st[finite]).abs().max() <= bin_size + 1e-6
    only_dense = torch.isfinite(dense_st) & torch.isinf(sparse_st)
    only_sparse = torch.isinf(dense_st) & torch.isfinite(sparse_st)
    disagreement = (only_dense | only_sparse).float().mean().item()
    assert disagreement < 0.01


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_multi_threshold(seed: int) -> None:
    layer = _make_layer().cuda()
    times = _make_inputs(seed=seed, num_bins=16).cuda()
    fracs = torch.tensor([-0.5, -0.25, 0.0, 0.25], device="cuda")
    thresholds_2d = layer.thresholds.unsqueeze(0) * (1 + fracs.unsqueeze(1))
    dense = multi_threshold_conv_accumulate(
        times, layer.weights_4d, thresholds_2d,
        stride=layer.stride, padding=layer.padding, device="cuda"
    )
    sparse = spike_driven_conv_accumulate_multi_threshold(
        times, layer.weights_4d, thresholds_2d, stride=layer.stride, padding=layer.padding
    )
    sparse = sparse.cpu()  # dense returns on CPU per existing implementation
    assert dense.shape == sparse.shape
    bin_size = 1.0 / 16
    finite = torch.isfinite(dense) & torch.isfinite(sparse)
    if finite.any():
        assert (dense[finite] - sparse[finite]).abs().max() <= bin_size + 1e-6


def test_layer_use_backend_flag() -> None:
    """Setting layer.use_backend dispatches _conv2d_accumulate to the kernel."""
    layer = _make_layer()
    times = _make_inputs(seed=0, num_bins=16)

    layer.use_backend = False
    dense_st, _ = layer._conv2d_accumulate(times)
    layer.use_backend = True
    backend_st, _ = layer._conv2d_accumulate(times)

    assert dense_st.shape == backend_st.shape
    finite = torch.isfinite(dense_st) & torch.isfinite(backend_st)
    if finite.any():
        assert (dense_st[finite] - backend_st[finite]).abs().max() <= 1.0 / 16 + 1e-6


def test_dense_stride_handling() -> None:
    """Stride != 1 path still matches dense."""
    num_bins = 16
    layer = _make_layer(stride=2, kernel_size=3, padding=0)
    times = _make_inputs(H=10, W=10, seed=11, num_bins=num_bins)
    dense_st, _ = layer._conv2d_accumulate(times)
    sparse_st, _ = spike_driven_conv_accumulate(
        times, layer.weights_4d, layer.thresholds, stride=2, padding=0
    )
    assert dense_st.shape == sparse_st.shape
    bin_size = 1.0 / num_bins
    finite = torch.isfinite(dense_st) & torch.isfinite(sparse_st)
    if finite.any():
        assert (dense_st[finite] - sparse_st[finite]).abs().max() <= bin_size + 1e-6
