import torch
import pytest

from spiking.preprocessing.whitening_kernels import (
    fit_whitening_kernels,
    apply_whitening_kernels,
)


def make_random_rgb_images(n=100, h=32, w=32):
    """Create random RGB images in [0, 1] for testing."""
    return torch.rand(n, 3, h, w)


class TestFitWhiteningKernels:
    def test_kernel_shape_rgb(self):
        images = make_random_rgb_images(n=50)
        kernels, mean = fit_whitening_kernels(images, patch_size=5, n_patches=500)
        # Cross-channel kernels: (C_out, C_in, kH, kW)
        assert kernels.shape == (3, 3, 5, 5)

    def test_kernel_shape_grayscale(self):
        images = torch.rand(50, 1, 16, 16)
        kernels, mean = fit_whitening_kernels(images, patch_size=3, n_patches=200)
        assert kernels.shape == (1, 1, 3, 3)

    def test_default_rho(self):
        """Default rho=1.0 per Falez 2020 Table I; all eigenvalues retained."""
        images = make_random_rgb_images(n=50)
        kernels, mean = fit_whitening_kernels(images, patch_size=5, n_patches=500)
        # With rho=1.0, all eigenvalues retained; kernel shape unchanged
        assert kernels.shape == (3, 3, 5, 5)

    def test_filter_mean_is_zero(self):
        """Each kernel should have zero mean (DC removal)."""
        images = make_random_rgb_images(n=50)
        kernels, mean = fit_whitening_kernels(images, patch_size=5, n_patches=500)
        for c in range(kernels.shape[0]):
            assert abs(kernels[c].mean().item()) < 1e-6

    def test_mean_shape(self):
        images = make_random_rgb_images(n=50)
        kernels, mean = fit_whitening_kernels(images, patch_size=5, n_patches=500)
        # Mean is per-pixel across patches: (C * kH * kW,)
        assert mean.shape == (3 * 5 * 5,)

    def test_kernels_are_finite(self):
        images = make_random_rgb_images(n=50)
        kernels, mean = fit_whitening_kernels(images, patch_size=5, n_patches=500)
        assert torch.isfinite(kernels).all()
        assert torch.isfinite(mean).all()

    def test_rho_reduces_eigenvalues(self):
        """Using rho < 1 should still produce valid kernels."""
        images = make_random_rgb_images(n=50)
        kernels, mean = fit_whitening_kernels(
            images, patch_size=5, n_patches=500, rho=0.5
        )
        assert kernels.shape == (3, 3, 5, 5)
        assert torch.isfinite(kernels).all()


class TestApplyWhiteningKernels:
    def test_output_shape_preserved(self):
        images = make_random_rgb_images(n=10)
        kernels, mean = fit_whitening_kernels(images, patch_size=5, n_patches=500)
        whitened = apply_whitening_kernels(images, kernels, mean)
        assert whitened.shape == images.shape

    def test_single_image(self):
        images = make_random_rgb_images(n=20)
        kernels, mean = fit_whitening_kernels(images, patch_size=5, n_patches=500)
        single = images[:1]
        whitened = apply_whitening_kernels(single, kernels, mean)
        assert whitened.shape == single.shape

    def test_output_approximately_zero_mean(self):
        torch.manual_seed(42)
        images = make_random_rgb_images(n=200, h=16, w=16)
        kernels, mean = fit_whitening_kernels(images, patch_size=5, n_patches=5000)
        whitened = apply_whitening_kernels(images, kernels, mean)
        channel_means = whitened.mean(dim=(0, 2, 3))
        # Whitened data should have approximately zero mean per channel
        assert torch.abs(channel_means).max() < 0.5

    def test_output_is_finite(self):
        images = make_random_rgb_images(n=20)
        kernels, mean = fit_whitening_kernels(images, patch_size=5, n_patches=500)
        whitened = apply_whitening_kernels(images, kernels, mean)
        assert torch.isfinite(whitened).all()
