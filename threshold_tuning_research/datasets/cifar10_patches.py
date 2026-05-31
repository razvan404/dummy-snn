import json

import torch


class Cifar10PatchDataset:
    """Deterministic round-based patch iteration; each round shuffles and sees each image once."""

    def __init__(self, processed_dir: str):
        data = torch.load(f"{processed_dir}/train.pt", weights_only=True)
        self.images = data["images"]  # (N, C, H, W)
        self.labels = data["labels"]  # (N,)
        self.patch_positions = data["patch_positions"]  # (N, R, 2)
        self.round_orders = data["round_orders"]  # (R, N)

        self.kernels = torch.load(f"{processed_dir}/kernels.pt", weights_only=True)
        self.mean = torch.load(f"{processed_dir}/mean.pt", weights_only=True)

        with open(f"{processed_dir}/config.json") as f:
            config = json.load(f)
        self.patch_size = config["patch_size"]

    @property
    def num_rounds(self) -> int:
        return self.round_orders.shape[0]

    @property
    def num_images(self) -> int:
        return self.images.shape[0]

    @property
    def image_shape(self) -> tuple[int, int, int]:
        channels = self.images.shape[1]
        return (channels, self.patch_size, self.patch_size)

    def get_patch(
        self, round_idx: int, position_in_round: int
    ) -> tuple[torch.Tensor, int]:
        img_idx = self.round_orders[round_idx, position_in_round].item()
        row = self.patch_positions[img_idx, round_idx, 0].item()
        col = self.patch_positions[img_idx, round_idx, 1].item()
        ps = self.patch_size
        patch = self.images[img_idx, :, row : row + ps, col : col + ps].flatten()
        label = self.labels[img_idx]
        return patch, label
