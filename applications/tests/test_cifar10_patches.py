import json

import torch
import pytest


@pytest.fixture
def processed_dir(tmp_path):
    """Create a small fake processed CIFAR-10 dataset on disk."""
    N = 10
    C = 6  # 3 RGB * 2 (pos/neg)
    H, W = 32, 32
    patch_size = 5
    patches_per_image = 3

    images = torch.rand(N, C, H, W)
    labels = torch.arange(N, dtype=torch.long)

    max_row = H - patch_size
    max_col = W - patch_size
    rng = torch.Generator().manual_seed(42)
    rows = torch.randint(0, max_row + 1, (N, patches_per_image), generator=rng)
    cols = torch.randint(0, max_col + 1, (N, patches_per_image), generator=rng)
    patch_positions = torch.stack([rows, cols], dim=-1)  # (N, P, 2)

    round_orders = torch.stack(
        [torch.randperm(N, generator=rng) for _ in range(patches_per_image)]
    )  # (P, N)

    d = str(tmp_path)
    torch.save(
        {
            "images": images,
            "labels": labels,
            "patch_positions": patch_positions,
            "round_orders": round_orders,
        },
        f"{d}/train.pt",
    )

    torch.save(
        {
            "images": torch.rand(5, C, H, W),
            "labels": torch.arange(5, dtype=torch.long),
        },
        f"{d}/test.pt",
    )

    torch.save(torch.rand(3, 3, 9, 9), f"{d}/kernels.pt")
    torch.save(torch.rand(3 * 9 * 9), f"{d}/mean.pt")

    with open(f"{d}/config.json", "w") as f:
        json.dump(
            {
                "patch_size": patch_size,
                "patches_per_image": patches_per_image,
                "seed": 42,
            },
            f,
        )

    return d


class TestCifar10PatchDataset:
    def test_get_patch_shape(self, processed_dir):
        from applications.datasets.cifar10_patches import Cifar10PatchDataset

        ds = Cifar10PatchDataset(processed_dir)
        patch, label = ds.get_patch(0, 0)
        assert patch.shape == (6 * 5 * 5,)

    def test_num_rounds(self, processed_dir):
        from applications.datasets.cifar10_patches import Cifar10PatchDataset

        ds = Cifar10PatchDataset(processed_dir)
        assert ds.num_rounds == 3

    def test_num_images(self, processed_dir):
        from applications.datasets.cifar10_patches import Cifar10PatchDataset

        ds = Cifar10PatchDataset(processed_dir)
        assert ds.num_images == 10

    def test_image_shape(self, processed_dir):
        from applications.datasets.cifar10_patches import Cifar10PatchDataset

        ds = Cifar10PatchDataset(processed_dir)
        assert ds.image_shape == (6, 5, 5)

    def test_round_orders_are_permutations(self, processed_dir):
        from applications.datasets.cifar10_patches import Cifar10PatchDataset

        ds = Cifar10PatchDataset(processed_dir)
        for r in range(ds.num_rounds):
            order = ds.round_orders[r]
            assert set(order.tolist()) == set(range(ds.num_images))

    def test_round_orders_differ(self, processed_dir):
        from applications.datasets.cifar10_patches import Cifar10PatchDataset

        ds = Cifar10PatchDataset(processed_dir)
        orders = [ds.round_orders[r].tolist() for r in range(ds.num_rounds)]
        assert len(set(tuple(o) for o in orders)) > 1

    def test_get_patch_returns_correct_slice(self, processed_dir):
        from applications.datasets.cifar10_patches import Cifar10PatchDataset

        ds = Cifar10PatchDataset(processed_dir)
        round_idx, pos = 1, 3
        img_idx = ds.round_orders[round_idx, pos].item()
        row = ds.patch_positions[img_idx, round_idx, 0].item()
        col = ds.patch_positions[img_idx, round_idx, 1].item()
        expected = ds.images[img_idx, :, row : row + 5, col : col + 5].flatten()

        patch, _ = ds.get_patch(round_idx, pos)
        assert torch.equal(patch, expected)

    def test_labels_match_image_order(self, processed_dir):
        from applications.datasets.cifar10_patches import Cifar10PatchDataset

        ds = Cifar10PatchDataset(processed_dir)
        for r in range(ds.num_rounds):
            for i in range(ds.num_images):
                _, label = ds.get_patch(r, i)
                img_idx = ds.round_orders[r, i].item()
                assert label == ds.labels[img_idx]

    def test_kernels_loaded(self, processed_dir):
        from applications.datasets.cifar10_patches import Cifar10PatchDataset

        ds = Cifar10PatchDataset(processed_dir)
        assert ds.kernels.shape == (3, 3, 9, 9)
        assert ds.mean.shape == (3 * 9 * 9,)

    def test_patch_positions_in_range(self, processed_dir):
        from applications.datasets.cifar10_patches import Cifar10PatchDataset

        ds = Cifar10PatchDataset(processed_dir)
        H, W = ds.images.shape[2], ds.images.shape[3]
        ps = ds.patch_size
        assert (ds.patch_positions[:, :, 0] >= 0).all()
        assert (ds.patch_positions[:, :, 0] <= H - ps).all()
        assert (ds.patch_positions[:, :, 1] >= 0).all()
        assert (ds.patch_positions[:, :, 1] <= W - ps).all()

    def test_every_image_seen_once_per_round(self, processed_dir):
        from applications.datasets.cifar10_patches import Cifar10PatchDataset

        ds = Cifar10PatchDataset(processed_dir)
        for r in range(ds.num_rounds):
            seen = set()
            for i in range(ds.num_images):
                _, label = ds.get_patch(r, i)
                img_idx = ds.round_orders[r, i].item()
                assert img_idx not in seen
                seen.add(img_idx)
            assert len(seen) == ds.num_images


class TestTransferWeights:
    def test_reshape_correctness(self):
        from spiking import (
            IntegrateAndFireLayer,
            ConvIntegrateAndFireLayer,
            NormalInitialization,
        )
        from applications.visual_learning.patch_train import transfer_weights

        init = NormalInitialization(avg_threshold=10.0, min_threshold=1.0, std_dev=0.1)
        fc = IntegrateAndFireLayer(150, 32, init, refractory_period=float("inf"))
        conv = ConvIntegrateAndFireLayer(6, 32, 5, threshold_initialization=init)

        transfer_weights(fc, conv)

        assert torch.equal(
            conv.weights.data,
            fc.weights.data.reshape(32, 6, 5, 5),
        )

    def test_thresholds_transferred(self):
        from spiking import (
            IntegrateAndFireLayer,
            ConvIntegrateAndFireLayer,
            NormalInitialization,
        )
        from applications.visual_learning.patch_train import transfer_weights

        init = NormalInitialization(avg_threshold=10.0, min_threshold=1.0, std_dev=0.1)
        fc = IntegrateAndFireLayer(150, 32, init, refractory_period=float("inf"))
        conv = ConvIntegrateAndFireLayer(6, 32, 5, threshold_initialization=init)

        transfer_weights(fc, conv)

        assert torch.equal(conv.thresholds.data, fc.thresholds.data)
