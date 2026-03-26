import torch
import pytest

from spiking.preprocessing.whitened_spike_encoding import encode_whitened_image


class TestEncodeWhitenedImage:
    def test_output_shape_rgb(self):
        """3-channel input produces 6-channel output (pos + neg per channel)."""
        image = torch.randn(3, 8, 8)
        result = encode_whitened_image(image)
        assert result.shape == (6, 8, 8)

    def test_output_shape_single_channel(self):
        image = torch.randn(1, 8, 8)
        result = encode_whitened_image(image)
        assert result.shape == (2, 8, 8)

    def test_positive_values_go_to_positive_channel(self):
        """A pixel with value > 0 should spike early in the positive channel."""
        image = torch.tensor([[[0.5]]])  # (1, 1, 1), positive
        result = encode_whitened_image(image)
        # Positive channel (index 0) should have finite spike time
        assert torch.isfinite(result[0, 0, 0])
        # Negative channel (index 1) should be inf (no spike)
        assert torch.isinf(result[1, 0, 0])

    def test_negative_values_go_to_negative_channel(self):
        """A pixel with value < 0 should spike early in the negative channel."""
        image = torch.tensor([[[-0.7]]])  # (1, 1, 1), negative
        result = encode_whitened_image(image)
        # Positive channel should be inf
        assert torch.isinf(result[0, 0, 0])
        # Negative channel should have finite spike time
        assert torch.isfinite(result[1, 0, 0])

    def test_zero_gives_inf_both_channels(self):
        """Zero intensity means no spike in either channel."""
        image = torch.tensor([[[0.0]]])
        result = encode_whitened_image(image)
        assert torch.isinf(result[0, 0, 0])
        assert torch.isinf(result[1, 0, 0])

    def test_brighter_spikes_earlier(self):
        """Larger absolute values should produce earlier spike times."""
        image = torch.tensor([[[0.3, 0.8]]])  # (1, 1, 2)
        result = encode_whitened_image(image)
        # 0.8 should spike earlier than 0.3 in positive channel
        assert result[0, 0, 1] < result[0, 0, 0]

    def test_spike_times_in_valid_range(self):
        """Finite spike times should be in [0, 1)."""
        torch.manual_seed(42)
        image = torch.randn(3, 16, 16)
        result = encode_whitened_image(image)
        finite = result[torch.isfinite(result)]
        assert (finite >= 0).all()
        assert (finite < 1).all()

    def test_channel_ordering_interleaved(self):
        """Channels should be interleaved [R+, R-, G+, G-, B+, B-]."""
        # Image with: R=+1, G=-1, B=+0.5
        image = torch.tensor(
            [
                [[1.0]],  # R positive
                [[-1.0]],  # G negative
                [[0.5]],  # B positive
            ]
        )  # (3, 1, 1)
        result = encode_whitened_image(image)
        # R+ at index 0: finite, R- at index 1: inf
        assert torch.isfinite(result[0, 0, 0])
        assert torch.isinf(result[1, 0, 0])
        # G+ at index 2: inf, G- at index 3: finite
        assert torch.isinf(result[2, 0, 0])
        assert torch.isfinite(result[3, 0, 0])
        # B+ at index 4: finite, B- at index 5: inf
        assert torch.isfinite(result[4, 0, 0])
        assert torch.isinf(result[5, 0, 0])

    def test_per_sample_scaling(self):
        """Per-sample min-max scaling: max maps to +1, min maps to -1."""
        image = torch.tensor([[[2.0, -1.0]]])  # (1, 1, 2), min=-1, max=2
        result = encode_whitened_image(image)
        # 2.0 → scaled to +1.0 → pos channel: 1.0 → spike time 0.0 (earliest)
        assert result[0, 0, 0] == 0.0
        # -1.0 → scaled to -1.0 → neg channel: 1.0 → spike time 0.0 (earliest)
        assert result[1, 0, 1] == 0.0
