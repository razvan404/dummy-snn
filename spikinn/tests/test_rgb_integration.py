import torch
import pytest

from spikinn.preprocessing.whitening_kernels import (
    fit_whitening_kernels,
    apply_whitening_kernels,
)
from spikinn.preprocessing.whitened_spike_encoding import encode_whitened_image
from spikinn.preprocessing import DiscretizeTimes
from spikinn.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer
from spikinn.threshold import ConstantInitialization


def make_random_rgb_images(n: int, h: int, w: int, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(n, 3, h, w)


def run_full_pipeline(
    images: torch.Tensor,
    kernel_size: int = 5,
    num_filters: int = 8,
    num_bins: int = 16,
    whitening_patch_size: int = 9,
) -> tuple[torch.Tensor, ConvIntegrateAndFireLayer]:
    N, C, H, W = images.shape

    # Step 1: Fit and apply whitening
    kernels, mean = fit_whitening_kernels(
        images,
        patch_size=whitening_patch_size,
        n_patches=min(1000, N * H * W),
    )
    whitened = apply_whitening_kernels(images, kernels, mean)

    # Step 2: Encode each image to spike times
    encoded = torch.stack([encode_whitened_image(img) for img in whitened])

    # Step 3: Discretize
    spike_times = DiscretizeTimes(num_bins)(encoded)

    # Step 4: Conv inference
    in_channels = spike_times.shape[1]  # 2*C = 6
    iH, iW = spike_times.shape[2], spike_times.shape[3]
    init = ConstantInitialization(10.0)
    layer = ConvIntegrateAndFireLayer(
        in_channels=in_channels,
        num_filters=num_filters,
        kernel_size=kernel_size,
        threshold_initialization=init,
        refractory_period=float("inf"),
    )

    with torch.no_grad():
        output = layer.infer_spike_times_batch(spike_times)

    return output, layer


class TestFullPipeline:
    """End-to-end: raw RGB → whitening → spike encoding → conv inference."""

    def test_output_shape(self):
        images = make_random_rgb_images(8, 32, 32)
        output, layer = run_full_pipeline(images, kernel_size=5, num_filters=16)
        oH, oW = layer._compute_output_size(32, 32)
        assert output.shape == (8, 16, oH, oW)

    def test_output_contains_spikes(self):
        images = make_random_rgb_images(8, 32, 32)
        output, _ = run_full_pipeline(images, kernel_size=5, num_filters=16)
        # At least some neurons should spike
        assert torch.isfinite(output).any()

    def test_high_threshold_produces_non_spikinn(self):
        images = make_random_rgb_images(8, 32, 32)
        output, layer = run_full_pipeline(images, kernel_size=5, num_filters=16)
        layer.thresholds.data.fill_(1000.0)
        with torch.no_grad():
            # Re-encode for this test
            kernels, mean = fit_whitening_kernels(images, patch_size=9, n_patches=500)
            whitened = apply_whitening_kernels(images, kernels, mean)
            encoded = torch.stack([encode_whitened_image(img) for img in whitened])
            spike_times = DiscretizeTimes(16)(encoded)
            output = layer.infer_spike_times_batch(spike_times)
        assert torch.isinf(output).any()

    def test_spike_times_non_negative(self):
        images = make_random_rgb_images(8, 32, 32)
        output, _ = run_full_pipeline(images, kernel_size=5, num_filters=16)
        finite = output[torch.isfinite(output)]
        assert (finite >= 0).all()


class TestWhiteningToSpikes:
    """Validate intermediate stages of the pipeline."""

    def test_whitening_preserves_shape(self):
        images = make_random_rgb_images(16, 32, 32)
        kernels, mean = fit_whitening_kernels(images, patch_size=9, n_patches=500)
        whitened = apply_whitening_kernels(images, kernels, mean)
        assert whitened.shape == images.shape

    def test_encoding_doubles_channels(self):
        images = make_random_rgb_images(4, 32, 32)
        kernels, mean = fit_whitening_kernels(images, patch_size=9, n_patches=500)
        whitened = apply_whitening_kernels(images, kernels, mean)
        encoded = encode_whitened_image(whitened[0])
        assert encoded.shape == (6, 32, 32)  # 2 * 3 channels

    def test_discretization_bins(self):
        images = make_random_rgb_images(4, 32, 32)
        kernels, mean = fit_whitening_kernels(images, patch_size=9, n_patches=500)
        whitened = apply_whitening_kernels(images, kernels, mean)
        encoded = encode_whitened_image(whitened[0])
        discretized = DiscretizeTimes(16)(encoded)
        finite = discretized[torch.isfinite(discretized)]
        # All finite values should be multiples of 1/16
        remainders = (finite * 16) - torch.floor(finite * 16)
        assert torch.allclose(remainders, torch.zeros_like(remainders), atol=1e-6)

    def test_spike_times_in_valid_range(self):
        images = make_random_rgb_images(4, 32, 32)
        kernels, mean = fit_whitening_kernels(images, patch_size=9, n_patches=500)
        whitened = apply_whitening_kernels(images, kernels, mean)
        encoded = encode_whitened_image(whitened[0])
        finite = encoded[torch.isfinite(encoded)]
        assert (finite >= 0).all()
        assert (finite < 1).all()


class TestConvBatchVsSingleOnRealData:
    """Verify batch and single-sample inference match on realistic spike-encoded data."""

    def test_single_vs_batch_on_whitened(self):
        images = make_random_rgb_images(4, 32, 32, seed=11)
        kernels, mean = fit_whitening_kernels(images, patch_size=9, n_patches=500)
        whitened = apply_whitening_kernels(images, kernels, mean)
        encoded = torch.stack([encode_whitened_image(img) for img in whitened])
        spike_times = DiscretizeTimes(16)(encoded)

        iH, iW = spike_times.shape[2], spike_times.shape[3]
        init = ConstantInitialization(10.0)
        layer = ConvIntegrateAndFireLayer(
            in_channels=6,
            num_filters=8,
            kernel_size=5,
            threshold_initialization=init,
            refractory_period=float("inf"),
        )

        with torch.no_grad():
            batch_result = layer.infer_spike_times_batch(spike_times)
            for i in range(len(spike_times)):
                single_result = layer.infer_spike_times(spike_times[i])
                torch.testing.assert_close(
                    batch_result[i],
                    single_result,
                    msg=f"Mismatch at image {i}",
                )
