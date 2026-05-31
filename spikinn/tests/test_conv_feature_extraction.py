import torch

from spikinn.evaluation.conv_feature_extraction import sum_pool_features


class TestSumPoolFeatures:
    def test_output_shape_quadrant(self):
        features = torch.rand(4, 8, 8)
        pooled = sum_pool_features(features, pool_size=2)
        assert pooled.shape == (4, 2, 2)

    def test_output_shape_pool4(self):
        features = torch.rand(4, 8, 8)
        pooled = sum_pool_features(features, pool_size=4)
        assert pooled.shape == (4, 4, 4)

    def test_quadrant_sum_values(self):
        features = torch.ones(1, 4, 4)
        pooled = sum_pool_features(features, pool_size=2)
        assert pooled.shape == (1, 2, 2)
        torch.testing.assert_close(pooled, torch.full((1, 2, 2), 4.0))

    def test_quadrant_sum_distinct_values(self):
        features = torch.zeros(1, 4, 4)
        features[0, :2, :2] = 1.0  # top-left quadrant = 4 ones
        features[0, :2, 2:] = 2.0  # top-right quadrant = 4 twos
        features[0, 2:, :2] = 3.0  # bottom-left = 4 threes
        features[0, 2:, 2:] = 4.0  # bottom-right = 4 fours
        pooled = sum_pool_features(features, pool_size=2)
        expected = torch.tensor([[[4.0, 8.0], [12.0, 16.0]]])
        torch.testing.assert_close(pooled, expected)

    def test_batched_input(self):
        features = torch.ones(3, 4, 8, 8)
        pooled = sum_pool_features(features, pool_size=2)
        assert pooled.shape == (3, 4, 2, 2)
        # Each 4×4 region of ones sums to 16.0
        torch.testing.assert_close(pooled, torch.full((3, 4, 2, 2), 16.0))

    def test_pool_size_1_is_identity(self):
        features = torch.rand(4, 6, 6)
        pooled = sum_pool_features(features, pool_size=1)
        torch.testing.assert_close(pooled, features)
