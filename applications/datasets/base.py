import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from spiking.preprocessing import (
    apply_difference_of_gaussians_filter_batch,
    apply_latency_encoding,
    discretize_times,
)


class SpikeEncodingDataset(Dataset):
    """Spike-encoded image dataset with optional random patch extraction.

    When patch_size and num_patches are set, __getitem__ returns a tensor of
    patches instead of the full image. Patch positions are sampled uniformly
    (distinct per image) and cached for reproducibility.

    :param inputs: (N, H, W) raw images.
    :param outputs: (N,) integer labels.
    :param image_shape: Optional resize target (H, W).
    :param cache_path: Path for caching spike times (and patch positions).
    :param patch_size: Side length of square patches. None to disable.
    :param num_patches: Number of patches per image. 0 to disable.
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        image_shape: tuple[int, int] | None = None,
        cache_path: str | None = None,
        patch_size: int | None = None,
        num_patches: int = 0,
    ):
        self.outputs = outputs.long()
        self.inputs = inputs.float()

        if image_shape is not None:
            self.inputs = F.interpolate(
                self.inputs.unsqueeze(1),
                size=image_shape,
                mode="nearest",
            ).squeeze(1)

        if cache_path and os.path.exists(cache_path):
            self.all_times = torch.load(cache_path, weights_only=True)
        else:
            dog = apply_difference_of_gaussians_filter_batch(self.inputs)
            self.all_times = discretize_times(apply_latency_encoding(dog))
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(self.all_times, cache_path)

        # Patch extraction setup
        self._patch_size = patch_size
        self._num_patches = num_patches
        self._patch_positions = None

        if patch_size is not None and num_patches > 0:
            patch_cache = (
                f"{cache_path}_patches_{patch_size}x{patch_size}_{num_patches}.pt"
                if cache_path
                else None
            )
            if patch_cache and os.path.exists(patch_cache):
                self._patch_positions = torch.load(patch_cache, weights_only=True)
            else:
                C_and_spatial = self.all_times.shape
                H, W = C_and_spatial[-2], C_and_spatial[-1]
                N = len(self.inputs)
                self._patch_positions = self._generate_patch_positions(
                    N, H, W, patch_size, num_patches
                )
                if patch_cache:
                    torch.save(self._patch_positions, patch_cache)

    @staticmethod
    def _generate_patch_positions(
        N: int, H: int, W: int, patch_size: int, num_patches: int
    ) -> torch.Tensor:
        """Generate distinct uniform patch positions for each image.

        :param N: Number of images.
        :param H: Spatial height of encoded images.
        :param W: Spatial width of encoded images.
        :param patch_size: Patch side length.
        :param num_patches: Patches per image.
        :returns: (N, num_patches, 2) int tensor of (row, col) positions.
        """
        max_row = H - patch_size
        max_col = W - patch_size
        num_valid = (max_row + 1) * (max_col + 1)

        positions = torch.zeros(N, num_patches, 2, dtype=torch.long)
        for i in range(N):
            indices = torch.randperm(num_valid)[:num_patches]
            positions[i, :, 0] = indices // (max_col + 1)
            positions[i, :, 1] = indices % (max_col + 1)
        return positions

    @property
    def patch_positions(self) -> torch.Tensor | None:
        """Cached (N, num_patches, 2) patch positions, or None if not patching."""
        return self._patch_positions

    @property
    def image_shape(self) -> tuple[int, int]:
        """Return (H, W) of the stored images."""
        return (self.inputs.shape[1], self.inputs.shape[2])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        if self._patch_positions is not None:
            times = self.all_times[idx]  # (C, H, W)
            ps = self._patch_size
            positions = self._patch_positions[idx]  # (num_patches, 2)
            patches = torch.stack(
                [times[:, r : r + ps, c : c + ps] for r, c in positions]
            )  # (num_patches, C, patch_size, patch_size)
            return patches, self.outputs[idx]
        return self.all_times[idx], self.outputs[idx]
