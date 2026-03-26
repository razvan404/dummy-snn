import torch
import numpy as np
import pytest
from torch.utils.data import DataLoader, TensorDataset

from spiking.evaluation.conv_feature_extraction import (
    sum_pool_features,
    extract_conv_features,
)
from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer
from spiking.threshold import ConstantInitialization


def make_layer(in_channels=6, num_filters=4, kernel_size=5, padding=0, threshold=5.0):
    init = ConstantInitialization(threshold)
    return ConvIntegrateAndFireLayer(
        in_channels=in_channels,
        num_filters=num_filters,
        kernel_size=kernel_size,
        padding=padding,
        threshold_initialization=init,
    )


class TestSumPoolFeatures:
    def test_output_shape_quadrant(self):
        """pool_size=2 divides 8×8 into 2×2 grid of 4×4 regions → output (4, 2, 2)."""
        features = torch.rand(4, 8, 8)
        pooled = sum_pool_features(features, pool_size=2)
        assert pooled.shape == (4, 2, 2)

    def test_output_shape_pool4(self):
        """pool_size=4 divides 8×8 into 4×4 grid of 2×2 regions → output (4, 4, 4)."""
        features = torch.rand(4, 8, 8)
        pooled = sum_pool_features(features, pool_size=4)
        assert pooled.shape == (4, 4, 4)

    def test_quadrant_sum_values(self):
        """Each region sums its elements. 4×4 of ones, pool_size=2 → 2×2 regions of size 2×2, each sums to 4."""
        features = torch.ones(1, 4, 4)
        pooled = sum_pool_features(features, pool_size=2)
        assert pooled.shape == (1, 2, 2)
        torch.testing.assert_close(pooled, torch.full((1, 2, 2), 4.0))

    def test_quadrant_sum_distinct_values(self):
        """Verify each quadrant sums independently."""
        features = torch.zeros(1, 4, 4)
        features[0, :2, :2] = 1.0  # top-left quadrant = 4 ones
        features[0, :2, 2:] = 2.0  # top-right quadrant = 4 twos
        features[0, 2:, :2] = 3.0  # bottom-left = 4 threes
        features[0, 2:, 2:] = 4.0  # bottom-right = 4 fours
        pooled = sum_pool_features(features, pool_size=2)
        expected = torch.tensor([[[4.0, 8.0], [12.0, 16.0]]])
        torch.testing.assert_close(pooled, expected)

    def test_batched_input(self):
        """Works with (B, F, H, W) batched input."""
        features = torch.ones(3, 4, 8, 8)
        pooled = sum_pool_features(features, pool_size=2)
        assert pooled.shape == (3, 4, 2, 2)
        # Each 4×4 region of ones sums to 16.0
        torch.testing.assert_close(pooled, torch.full((3, 4, 2, 2), 16.0))

    def test_pool_size_1_is_identity(self):
        features = torch.rand(4, 6, 6)
        pooled = sum_pool_features(features, pool_size=1)
        torch.testing.assert_close(pooled, features)


class TestExtractConvFeatures:
    def test_output_shapes(self):
        layer = make_layer(in_channels=2, num_filters=4, kernel_size=3, threshold=0.5)
        layer.weights.data.fill_(0.5)

        n = 10
        times = torch.rand(n, 2, 8, 8)
        times[times > 0.7] = float("inf")
        labels = torch.randint(0, 10, (n,))
        dataset = TensorDataset(times, labels)
        loader = DataLoader(dataset, batch_size=None, shuffle=False)

        X, y = extract_conv_features(layer, loader, pool_size=2)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == n
        assert y.shape == (n,)
        # oH = oW = 8-3+1 = 6, pool_size=2 → 2×2 grid, flattened: 4*2*2 = 16
        assert X.shape[1] == 4 * 2 * 2

    def test_features_in_valid_range(self):
        layer = make_layer(in_channels=2, num_filters=4, kernel_size=3, threshold=0.5)
        layer.weights.data.fill_(0.5)

        n = 5
        times = torch.rand(n, 2, 8, 8)
        times[times > 0.7] = float("inf")
        labels = torch.randint(0, 10, (n,))
        dataset = TensorDataset(times, labels)
        loader = DataLoader(dataset, batch_size=None, shuffle=False)

        X, y = extract_conv_features(layer, loader, pool_size=2)
        assert (X >= 0).all()

    def test_with_t_target(self):
        layer = make_layer(in_channels=2, num_filters=4, kernel_size=3, threshold=0.5)
        layer.weights.data.fill_(0.5)

        n = 5
        times = torch.rand(n, 2, 8, 8)
        times[times > 0.7] = float("inf")
        labels = torch.randint(0, 10, (n,))
        dataset = TensorDataset(times, labels)
        loader = DataLoader(dataset, batch_size=None, shuffle=False)

        X, y = extract_conv_features(layer, loader, pool_size=2, t_target=0.5)
        assert X.shape[0] == n

    def test_pool_size_1(self):
        layer = make_layer(in_channels=2, num_filters=4, kernel_size=3, threshold=0.5)
        layer.weights.data.fill_(0.5)

        n = 5
        times = torch.rand(n, 2, 8, 8)
        times[times > 0.7] = float("inf")
        labels = torch.randint(0, 10, (n,))
        dataset = TensorDataset(times, labels)
        loader = DataLoader(dataset, batch_size=None, shuffle=False)

        X, y = extract_conv_features(layer, loader, pool_size=1)
        # No pooling: 4 * 6 * 6 = 144
        assert X.shape[1] == 4 * 6 * 6
