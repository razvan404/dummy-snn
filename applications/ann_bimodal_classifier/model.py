"""Bounded-encoder + small classifier head for CIFAR-10.

The encoder mirrors ``applications.ann_bimodal_autoencoder``: a single Conv2d
with weights clipped to [0, 1] and a smooth bimodal penalty applied. The rest
of the network (BatchNorm → adaptive max-pool to 2×2 → linear head) is
unconstrained.
"""

import torch
import torch.nn as nn


class ConvBimodalClassifier(nn.Module):
    """[0,1]-bounded conv encoder + 2×2 max-pool + linear classifier.

    :param in_channels: 3 for raw RGB, 6 for whitened-spike intensity.
    :param num_filters: Encoder feature count (default 256).
    :param kernel_size: Square kernel side (default 5).
    :param num_classes: Output classes (default 10 for CIFAR-10).
    :param pool_out: Spatial output of the adaptive max-pool (default 2).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_filters: int = 256,
        kernel_size: int = 5,
        num_classes: int = 10,
        pool_out: int = 2,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.encoder = nn.Conv2d(
            in_channels,
            num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(num_filters)
        self.pool = nn.AdaptiveMaxPool2d((pool_out, pool_out))
        self.classifier = nn.Linear(num_filters * pool_out * pool_out, num_classes)

        with torch.no_grad():
            self.encoder.weight.uniform_(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.encoder(x))
        h = self.norm(h)
        h = self.pool(h)
        h = h.flatten(1)
        return self.classifier(h)

    @torch.no_grad()
    def clip_encoder_weights(self, lo: float = 0.0, hi: float = 1.0) -> None:
        """Hard-clip encoder weights in place after each optimizer step."""
        self.encoder.weight.data.clamp_(lo, hi)
