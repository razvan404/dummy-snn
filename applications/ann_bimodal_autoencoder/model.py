"""Convolutional autoencoder with a bounded encoder and a free decoder.

Encoder weights are constrained to [0, 1] (mimicking STDP's bounded weight
range) so that a bimodal regularizer applied to them has a well-defined target.
The decoder is unconstrained so it retains full capacity to reconstruct.
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Single-layer conv autoencoder.

    :param in_channels: Input channel count (3 for RGB; 6 for whitened spike-time).
    :param out_channels: Reconstruction target channel count (defaults to in_channels).
    :param num_filters: Encoder feature count (default 256).
    :param kernel_size: Square kernel side (default 5).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int | None = None,
        num_filters: int = 256,
        kernel_size: int = 5,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        padding = kernel_size // 2

        self.encoder = nn.Conv2d(
            in_channels,
            num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        # Sits downstream of the bounded encoder; rescales the large activations
        # produced by U(0,1)-bounded weights so the decoder sigmoid does not
        # saturate. Considered part of the decoding pipeline (free, unconstrained).
        self.norm = nn.BatchNorm2d(num_filters)
        self.decoder = nn.ConvTranspose2d(
            num_filters,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )

        with torch.no_grad():
            self.encoder.weight.uniform_(0.0, 1.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(self.norm(h)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @torch.no_grad()
    def clip_encoder_weights(self, lo: float = 0.0, hi: float = 1.0) -> None:
        """Hard-clip encoder weights in place after each optimizer step."""
        self.encoder.weight.data.clamp_(lo, hi)


def bimodal_penalty(weights: torch.Tensor) -> torch.Tensor:
    """Mean of min(w**2, (w-1)**2): zero at {0,1}, max 0.25 at w=0.5."""
    return torch.minimum(weights.pow(2), (weights - 1.0).pow(2)).mean()


def bimodality_score(weights: torch.Tensor, tol: float = 0.1) -> float:
    """Fraction of weights within `tol` of {0, 1}. Cheap bimodality proxy."""
    w = weights.detach()
    near_extreme = (w.abs() <= tol) | ((w - 1.0).abs() <= tol)
    return near_extreme.float().mean().item()
