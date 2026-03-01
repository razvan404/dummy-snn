import pytest
import torch

from spiking.preprocessing import discretize_times


class TestDiscretizeTimes:
    def test_finite_values_quantized_to_bin_edges(self):
        """Values in [0, 1) should snap to floor(v * num_bins) / num_bins."""
        times = torch.tensor([0.0, 0.003, 0.5, 0.999])
        result = discretize_times(times, num_bins=256)

        expected = torch.tensor(
            [
                0.0 / 256,  # floor(0.0 * 256) / 256 = 0
                0.0 / 256,  # floor(0.003 * 256) / 256 = floor(0.768) / 256 = 0
                128.0 / 256,  # floor(0.5 * 256) / 256 = 128/256
                255.0
                / 256,  # floor(0.999 * 256) / 256 = floor(255.744) / 256 = 255/256
            ]
        )
        assert torch.allclose(result, expected)

    def test_inf_preserved(self):
        """Infinite values (no spike) must remain inf."""
        times = torch.tensor([0.1, float("inf"), 0.5, float("inf")])
        result = discretize_times(times, num_bins=256)

        assert torch.isinf(result[1])
        assert torch.isinf(result[3])
        assert torch.isfinite(result[0])
        assert torch.isfinite(result[2])

    def test_idempotent(self):
        """Applying discretize_times twice gives the same result as once."""
        times = torch.tensor([0.0, 0.123, 0.456, 0.789, float("inf")])
        once = discretize_times(times, num_bins=256)
        twice = discretize_times(once, num_bins=256)

        assert torch.equal(once[:4], twice[:4])
        assert torch.isinf(twice[4])

    def test_does_not_modify_input(self):
        """discretize_times must not mutate the input tensor."""
        times = torch.tensor([0.25, 0.75, float("inf")])
        original = times.clone()
        discretize_times(times, num_bins=256)

        assert torch.equal(times, original)

    def test_custom_num_bins(self):
        """Different bin counts should produce different quantizations."""
        times = torch.tensor([0.15])
        result_10 = discretize_times(times, num_bins=10)
        result_256 = discretize_times(times, num_bins=256)

        # floor(0.15 * 10) / 10 = 1/10 = 0.1
        assert result_10.item() == pytest.approx(0.1)
        # floor(0.15 * 256) / 256 = floor(38.4) / 256 = 38/256
        assert result_256.item() == pytest.approx(38.0 / 256)
