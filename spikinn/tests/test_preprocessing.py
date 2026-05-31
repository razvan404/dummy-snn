import pytest
import torch

from spikinn.preprocessing import DiscretizeTimes


class TestDifferenceOfGaussians:
    def test_output_shape_and_channels(self):
        from spikinn.preprocessing.difference_of_gaussians import (
            apply_difference_of_gaussians_filter,
        )

        image = torch.rand(8, 8)
        result = apply_difference_of_gaussians_filter(image)
        assert result.shape == (2, 8, 8)

    def test_on_off_channels_non_negative(self):
        from spikinn.preprocessing.difference_of_gaussians import (
            apply_difference_of_gaussians_filter,
        )

        image = torch.rand(8, 8)
        result = apply_difference_of_gaussians_filter(image)
        assert (result >= 0).all()

    def test_deterministic(self):
        from spikinn.preprocessing.difference_of_gaussians import (
            apply_difference_of_gaussians_filter,
        )

        image = torch.rand(8, 8)
        r1 = apply_difference_of_gaussians_filter(image)
        r2 = apply_difference_of_gaussians_filter(image)
        assert torch.equal(r1, r2)


class TestDiscretizeTimes:
    def test_finite_values_quantized_to_bin_edges(self):
        times = torch.tensor([0.0, 0.003, 0.5, 0.999])
        result = DiscretizeTimes(256)(times)

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
        times = torch.tensor([0.1, float("inf"), 0.5, float("inf")])
        result = DiscretizeTimes(256)(times)

        assert torch.isinf(result[1])
        assert torch.isinf(result[3])
        assert torch.isfinite(result[0])
        assert torch.isfinite(result[2])

    def test_idempotent(self):
        times = torch.tensor([0.0, 0.123, 0.456, 0.789, float("inf")])
        once = DiscretizeTimes(256)(times)
        twice = DiscretizeTimes(256)(once)

        assert torch.equal(once[:4], twice[:4])
        assert torch.isinf(twice[4])

    def test_does_not_modify_input(self):
        times = torch.tensor([0.25, 0.75, float("inf")])
        original = times.clone()
        DiscretizeTimes(256)(times)

        assert torch.equal(times, original)

    def test_custom_num_bins(self):
        times = torch.tensor([0.15])
        result_10 = DiscretizeTimes(10)(times)
        result_256 = DiscretizeTimes(256)(times)

        # floor(0.15 * 10) / 10 = 1/10 = 0.1
        assert result_10.item() == pytest.approx(0.1)
        # floor(0.15 * 256) / 256 = floor(38.4) / 256 = 38/256
        assert result_256.item() == pytest.approx(38.0 / 256)
