from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from spiking.tests.multi_threshold_reference import multi_threshold_conv_accumulate
from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer
from spiking.threshold.normal_initialization import NormalInitialization
import spiking_backend
from spiking_backend.reference import (
    first_spike_times_gather,
    first_spike_times_gather_multi_threshold,
    spike_driven_conv_accumulate,
    spike_driven_conv_accumulate_multi_threshold,
)


def classic_discrete_time_sim(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds: torch.Tensor,
    num_bins: int,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    B, C, H, W = input_times.shape
    F_, _, kH, kW = weights_4d.shape
    oH = (H + 2 * padding - kH) // stride + 1
    oW = (W + 2 * padding - kW) // stride + 1

    spike_times = torch.full(
        (B, F_, oH, oW), float("inf"), dtype=input_times.dtype, device=input_times.device
    )
    potential = torch.zeros((B, F_, oH, oW), dtype=input_times.dtype, device=input_times.device)
    th = thresholds.view(1, F_, 1, 1)

    for ti in range(num_bins):
        t_val = ti / num_bins
        spikes_t = (input_times == t_val).to(input_times.dtype)
        if spikes_t.any():
            contrib = F.conv2d(spikes_t, weights_4d, stride=stride, padding=padding)
            potential = potential + contrib
        not_yet = torch.isinf(spike_times)
        crossed = not_yet & (potential >= th)
        if crossed.any():
            spike_times = torch.where(
                crossed, torch.full_like(spike_times, t_val), spike_times
            )
    return spike_times


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
    only_dense = torch.isfinite(dense_st) & torch.isinf(sparse_st)
    only_sparse = torch.isinf(dense_st) & torch.isfinite(sparse_st)
    disagreement = (only_dense | only_sparse).float().mean().item()
    assert disagreement < 0.01, f"too many spike/no-spike mismatches: {disagreement}"


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
    sparse = sparse.cpu()
    assert dense.shape == sparse.shape
    bin_size = 1.0 / 16
    finite = torch.isfinite(dense) & torch.isfinite(sparse)
    if finite.any():
        assert (dense[finite] - sparse[finite]).abs().max() <= bin_size + 1e-6


def test_layer_backend_flag() -> None:
    layer = _make_layer()
    times = _make_inputs(seed=0, num_bins=16)
    bin_tol = 1.0 / 16 + 1e-6

    layer._backend = "dense"
    dense_st, _ = layer._conv2d_accumulate(times)

    for backend in ("scatter", "gather"):
        layer._backend = backend
        backend_st, _ = layer._conv2d_accumulate(times)
        assert dense_st.shape == backend_st.shape, backend
        finite = torch.isfinite(dense_st) & torch.isfinite(backend_st)
        if finite.any():
            assert (
                (dense_st[finite] - backend_st[finite]).abs().max() <= bin_tol
            ), backend


def test_layer_backend_default_is_gather() -> None:
    layer = _make_layer()
    assert layer._backend == "gather"


def test_layer_gather_returns_cum_potential_when_requested() -> None:
    layer = _make_layer()
    times = _make_inputs(seed=0, num_bins=16)

    layer._backend = "gather"
    _, pot_skip = layer._conv2d_accumulate(times, with_cum_potential=False)
    assert pot_skip.numel() == 0

    _, pot_gather = layer._conv2d_accumulate(times, with_cum_potential=True)
    assert pot_gather.shape == (times.shape[0], 16, 8, 8)

    layer._backend = "scatter"
    _, pot_scatter = layer._conv2d_accumulate(times, with_cum_potential=True)
    assert torch.allclose(pot_gather, pot_scatter, rtol=1e-4, atol=1e-4)


def test_layer_backend_invalid_name_raises() -> None:
    layer = _make_layer()
    times = _make_inputs(seed=0, num_bins=16)
    layer._backend = "nonexistent"
    with pytest.raises(ValueError, match="unknown backend"):
        layer._conv2d_accumulate(times)


def test_layer_constructor_accepts_backend() -> None:
    init = NormalInitialization(avg_threshold=2.0, std_dev=0.2, min_threshold=0.5)
    default = ConvIntegrateAndFireLayer(
        in_channels=2, num_filters=4, kernel_size=3,
        threshold_initialization=init, refractory_period=float("inf"),
    )
    assert default._backend == "gather"

    explicit = ConvIntegrateAndFireLayer(
        in_channels=2, num_filters=4, kernel_size=3,
        threshold_initialization=init, refractory_period=float("inf"),
        backend="scatter",
    )
    assert explicit._backend == "scatter"

    with pytest.raises(ValueError, match="unknown backend"):
        ConvIntegrateAndFireLayer(
            in_channels=2, num_filters=4, kernel_size=3,
            threshold_initialization=init, refractory_period=float("inf"),
            backend="nope",
        )


@pytest.mark.parametrize("seed", [0, 2, 11])
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("num_bins", [16, 64])
def test_gather_first_spike_matches_dense(
    seed: int, padding: int, num_bins: int
) -> None:
    layer = _make_layer(padding=padding)
    times = _make_inputs(seed=seed, num_bins=num_bins)
    dense_st, _ = layer._conv2d_accumulate(times)
    gather_st = spiking_backend.first_spike_times(
        times,
        layer.weights_4d,
        layer.thresholds,
        num_bins=num_bins,
        stride=layer.stride,
        padding=layer.padding,
    )
    assert dense_st.shape == gather_st.shape
    bin_size = 1.0 / num_bins
    finite = torch.isfinite(dense_st) & torch.isfinite(gather_st)
    if finite.any():
        assert (
            (dense_st[finite] - gather_st[finite]).abs().max() <= bin_size + 1e-6
        )
    only_dense = torch.isfinite(dense_st) & torch.isinf(gather_st)
    only_gather = torch.isinf(dense_st) & torch.isfinite(gather_st)
    disagreement = (only_dense | only_gather).float().mean().item()
    assert disagreement < 0.02, (
        f"too many spike/no-spike mismatches: {disagreement}"
    )


@pytest.mark.parametrize("seed", [0, 5])
@pytest.mark.parametrize("num_bins", [16, 64])
def test_gather_reference_matches_compiled_cpu(seed: int, num_bins: int) -> None:
    layer = _make_layer()
    times = _make_inputs(seed=seed, num_bins=num_bins)
    ref, _ = first_spike_times_gather(
        times,
        layer.weights_4d,
        layer.thresholds,
        num_bins=num_bins,
        stride=layer.stride,
        padding=layer.padding,
    )
    compiled = spiking_backend.first_spike_times(
        times,
        layer.weights_4d,
        layer.thresholds,
        num_bins=num_bins,
        stride=layer.stride,
        padding=layer.padding,
    )
    bin_size = 1.0 / num_bins
    finite = torch.isfinite(ref) & torch.isfinite(compiled)
    if finite.any():
        assert (ref[finite] - compiled[finite]).abs().max() <= bin_size + 1e-6


@pytest.mark.parametrize("seed", [1, 4])
@pytest.mark.parametrize("num_bins", [16, 64])
def test_gather_multi_threshold_cpu(seed: int, num_bins: int) -> None:
    layer = _make_layer()
    times = _make_inputs(seed=seed, num_bins=num_bins)
    fracs = torch.tensor([-0.5, -0.25, 0.0, 0.25])
    thresholds_2d = layer.thresholds.unsqueeze(0) * (1 + fracs.unsqueeze(1))

    ref, _ = first_spike_times_gather_multi_threshold(
        times,
        layer.weights_4d,
        thresholds_2d,
        num_bins=num_bins,
        stride=layer.stride,
        padding=layer.padding,
    )
    compiled = spiking_backend.first_spike_times_multi_threshold(
        times,
        layer.weights_4d,
        thresholds_2d,
        num_bins=num_bins,
        stride=layer.stride,
        padding=layer.padding,
    )
    bin_size = 1.0 / num_bins
    finite = torch.isfinite(ref) & torch.isfinite(compiled)
    if finite.any():
        assert (ref[finite] - compiled[finite]).abs().max() <= bin_size + 1e-6


def test_gather_wta_keeps_one_filter_per_position() -> None:
    layer = _make_layer()
    times = _make_inputs(seed=2, num_bins=16)
    out = spiking_backend.first_spike_times(
        times,
        layer.weights_4d,
        layer.thresholds,
        num_bins=16,
        stride=layer.stride,
        padding=layer.padding,
        wta=True,
    )
    finite_per_pos = torch.isfinite(out).sum(dim=1)  # (B, oH, oW)
    assert finite_per_pos.max().item() <= 1


def test_gather_wta_keeps_earliest_filter() -> None:
    layer = _make_layer()
    times = _make_inputs(seed=3, num_bins=16)
    base = spiking_backend.first_spike_times(
        times,
        layer.weights_4d,
        layer.thresholds,
        num_bins=16,
        stride=layer.stride,
        padding=layer.padding,
    )
    wta = spiking_backend.first_spike_times(
        times,
        layer.weights_4d,
        layer.thresholds,
        num_bins=16,
        stride=layer.stride,
        padding=layer.padding,
        wta=True,
    )
    expected_min, _ = base.min(dim=1)  # (B, oH, oW)
    actual_min, _ = wta.min(dim=1)
    finite = torch.isfinite(expected_min)
    if finite.any():
        assert torch.equal(actual_min[finite], expected_min[finite])


@pytest.mark.parametrize("seed", [0, 6])
@pytest.mark.parametrize("num_bins", [16, 64])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gather_cuda_matches_dense(seed: int, num_bins: int) -> None:
    layer = _make_layer().cuda()
    times = _make_inputs(seed=seed, num_bins=num_bins).cuda()
    dense_st, _ = layer._conv2d_accumulate(times)
    gather_st = spiking_backend.first_spike_times(
        times,
        layer.weights_4d,
        layer.thresholds,
        num_bins=num_bins,
        stride=layer.stride,
        padding=layer.padding,
    )
    assert dense_st.shape == gather_st.shape
    bin_size = 1.0 / num_bins
    finite = torch.isfinite(dense_st) & torch.isfinite(gather_st)
    if finite.any():
        assert (
            (dense_st[finite] - gather_st[finite]).abs().max() <= bin_size + 1e-6
        )
    only_dense = torch.isfinite(dense_st) & torch.isinf(gather_st)
    only_gather = torch.isinf(dense_st) & torch.isfinite(gather_st)
    disagreement = (only_dense | only_gather).float().mean().item()
    assert disagreement < 0.02


@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gather_cuda_multi_threshold(seed: int) -> None:
    layer = _make_layer().cuda()
    num_bins = 16
    times = _make_inputs(seed=seed, num_bins=num_bins).cuda()
    fracs = torch.tensor([-0.5, -0.25, 0.0, 0.25], device="cuda")
    thresholds_2d = layer.thresholds.unsqueeze(0) * (1 + fracs.unsqueeze(1))

    dense = multi_threshold_conv_accumulate(
        times,
        layer.weights_4d,
        thresholds_2d,
        stride=layer.stride,
        padding=layer.padding,
        device="cuda",
    )
    gather = spiking_backend.first_spike_times_multi_threshold(
        times,
        layer.weights_4d,
        thresholds_2d,
        num_bins=num_bins,
        stride=layer.stride,
        padding=layer.padding,
    ).cpu()  # dense returns on CPU per existing implementation
    bin_size = 1.0 / num_bins
    finite = torch.isfinite(dense) & torch.isfinite(gather)
    if finite.any():
        assert (dense[finite] - gather[finite]).abs().max() <= bin_size + 1e-6


@pytest.mark.parametrize("seed", [0, 3, 9])
@pytest.mark.parametrize("num_bins", [16, 64])
def test_backend_matches_classic_discrete_time_sim(
    seed: int, num_bins: int
) -> None:
    layer = _make_layer()
    times = _make_inputs(seed=seed, num_bins=num_bins)

    classic_st = classic_discrete_time_sim(
        times,
        layer.weights_4d,
        layer.thresholds,
        num_bins=num_bins,
        stride=layer.stride,
        padding=layer.padding,
    )

    gather_st = spiking_backend.first_spike_times(
        times,
        layer.weights_4d,
        layer.thresholds,
        num_bins=num_bins,
        stride=layer.stride,
        padding=layer.padding,
    )
    scatter_st, _ = spiking_backend.spike_driven_conv_accumulate(
        times,
        layer.weights_4d,
        layer.thresholds,
        stride=layer.stride,
        padding=layer.padding,
    )

    bin_size = 1.0 / num_bins
    finite_g = torch.isfinite(classic_st) & torch.isfinite(gather_st)
    if finite_g.any():
        assert (
            (classic_st[finite_g] - gather_st[finite_g]).abs().max() <= 1e-6
        ), "gather path drifted from classic discrete-time simulation"

    finite_s = torch.isfinite(classic_st) & torch.isfinite(scatter_st)
    if finite_s.any():
        assert (
            (classic_st[finite_s] - scatter_st[finite_s]).abs().max()
            <= bin_size + 1e-6
        )

    for name, backend_st in [("gather", gather_st), ("scatter", scatter_st)]:
        only_classic = torch.isfinite(classic_st) & torch.isinf(backend_st)
        only_backend = torch.isinf(classic_st) & torch.isfinite(backend_st)
        disagreement = (only_classic | only_backend).float().mean().item()
        assert disagreement < 0.02, (
            f"{name}: too many spike/no-spike mismatches vs classic sim: "
            f"{disagreement}"
        )


@pytest.mark.parametrize("seed", [0, 1, 4])
@pytest.mark.parametrize("padding", [0, 2])
def test_scatter_skip_spiked_matches_full(
    seed: int, padding: int
) -> None:
    layer = _make_layer(padding=padding)
    times = _make_inputs(seed=seed, num_bins=64)

    default_st, _ = spiking_backend.spike_driven_conv_accumulate(
        times,
        layer.weights_4d,
        layer.thresholds,
        stride=layer.stride,
        padding=layer.padding,
    )
    fast_st, _ = spiking_backend.spike_driven_conv_accumulate(
        times,
        layer.weights_4d,
        layer.thresholds,
        stride=layer.stride,
        padding=layer.padding,
        compute_cum_potential=False,
    )
    assert default_st.shape == fast_st.shape
    assert torch.equal(default_st, fast_st), (
        "fast mode produced different spike_times than default mode"
    )


@pytest.mark.parametrize("seed", [0, 2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_scatter_skip_spiked_cuda_matches_full(seed: int) -> None:
    layer = _make_layer().cuda()
    times = _make_inputs(seed=seed, num_bins=64).cuda()
    default_st, _ = spiking_backend.spike_driven_conv_accumulate(
        times, layer.weights_4d, layer.thresholds,
        stride=layer.stride, padding=layer.padding,
    )
    fast_st, _ = spiking_backend.spike_driven_conv_accumulate(
        times, layer.weights_4d, layer.thresholds,
        stride=layer.stride, padding=layer.padding,
        compute_cum_potential=False,
    )
    bin_size = 1.0 / 64
    finite = torch.isfinite(default_st) & torch.isfinite(fast_st)
    if finite.any():
        assert (
            (default_st[finite] - fast_st[finite]).abs().max()
            <= bin_size + 1e-6
        )
    only_default = torch.isfinite(default_st) & torch.isinf(fast_st)
    only_fast = torch.isinf(default_st) & torch.isfinite(fast_st)
    disagreement = (only_default | only_fast).float().mean().item()
    assert disagreement < 0.02


@pytest.mark.parametrize("num_bins", [16, 64])
@pytest.mark.parametrize("compute_cum_potential", [True, False])
def test_b1_patch_sized_input(num_bins: int, compute_cum_potential: bool) -> None:
    layer = _make_layer(kernel_size=5, padding=0)
    times = _make_inputs(batch=1, H=5, W=5, seed=0, num_bins=num_bins)
    dense_st, _ = layer._conv2d_accumulate(times)
    assert dense_st.shape == (1, 16, 1, 1)

    gather_st = spiking_backend.first_spike_times(
        times, layer.weights_4d, layer.thresholds,
        num_bins=num_bins, stride=layer.stride, padding=layer.padding,
    )
    scatter_st, _ = spiking_backend.spike_driven_conv_accumulate(
        times, layer.weights_4d, layer.thresholds,
        stride=layer.stride, padding=layer.padding,
        compute_cum_potential=compute_cum_potential,
    )
    assert gather_st.shape == (1, 16, 1, 1)
    assert scatter_st.shape == (1, 16, 1, 1)

    bin_size = 1.0 / num_bins
    for name, st in [("gather", gather_st), ("scatter", scatter_st)]:
        finite = torch.isfinite(dense_st) & torch.isfinite(st)
        if finite.any():
            assert (
                (dense_st[finite] - st[finite]).abs().max() <= bin_size + 1e-6
            ), f"{name} disagreed with dense at patch-sized input"


@pytest.mark.parametrize("num_bins", [16, 64])
def test_b1_patch_sized_multi_threshold(num_bins: int) -> None:
    layer = _make_layer(kernel_size=5, padding=0)
    times = _make_inputs(batch=1, H=5, W=5, seed=2, num_bins=num_bins)
    fracs = torch.tensor([-0.5, -0.25, 0.0, 0.25])
    thresholds_2d = layer.thresholds.unsqueeze(0) * (1 + fracs.unsqueeze(1))

    dense = multi_threshold_conv_accumulate(
        times, layer.weights_4d, thresholds_2d,
        stride=layer.stride, padding=layer.padding, device="cpu",
    )
    gather = spiking_backend.first_spike_times_multi_threshold(
        times, layer.weights_4d, thresholds_2d,
        num_bins=num_bins, stride=layer.stride, padding=layer.padding,
    )
    scatter = spiking_backend.spike_driven_conv_accumulate_multi_threshold(
        times, layer.weights_4d, thresholds_2d,
        stride=layer.stride, padding=layer.padding,
    )
    assert gather.shape == (4, 1, 16, 1, 1)
    assert scatter.shape == (4, 1, 16, 1, 1)

    bin_size = 1.0 / num_bins
    for name, st in [("gather", gather), ("scatter", scatter)]:
        finite = torch.isfinite(dense) & torch.isfinite(st)
        if finite.any():
            assert (dense[finite] - st[finite]).abs().max() <= bin_size + 1e-6, name


def test_dense_stride_handling() -> None:
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
