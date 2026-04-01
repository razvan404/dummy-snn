import torch
import torch.nn.functional as F


def _extract_patches(
    images: torch.Tensor, patch_size: int, n_patches: int
) -> torch.Tensor:
    """Extract random patches from images.

    Uses F.unfold for vectorized extraction rather than per-patch loops.

    Args:
        images: (N, C, H, W) tensor.
        patch_size: Side length of square patches.
        n_patches: Number of patches to extract.

    Returns:
        (n_patches, C * patch_size * patch_size) flattened patches.
    """
    N, C, H, W = images.shape
    dim = C * patch_size * patch_size

    # Unfold all patches: (N, C*kH*kW, L) where L = num_patches_per_image
    all_patches = F.unfold(images, kernel_size=patch_size)  # (N, dim, L)
    L = all_patches.shape[2]

    # Reshape to (N*L, dim)
    all_patches = all_patches.permute(0, 2, 1).reshape(-1, dim)

    # Randomly sample n_patches
    total = all_patches.shape[0]
    indices = torch.randint(0, total, (n_patches,))
    return all_patches[indices]


def compute_patch_mean(
    images: torch.Tensor,
    patch_size: int = 9,
    n_patches: int = 1_000_000,
) -> torch.Tensor:
    """Compute the mean patch vector from image patches.

    Args:
        images: (N, C, H, W) tensor of images in [0, 1].
        patch_size: Side length of square patches.
        n_patches: Number of random patches to sample.

    Returns:
        (C * patch_size * patch_size,) mean vector.
    """
    N, C, H, W = images.shape
    max_patches = N * (H - patch_size + 1) * (W - patch_size + 1)
    n_patches = min(n_patches, max_patches)
    patches = _extract_patches(images, patch_size, n_patches)
    return patches.mean(dim=0)


def load_kernels(path: str) -> torch.Tensor:
    """Load pre-computed whitening kernels from a .pt file.

    Args:
        path: Path to a .pt file containing a (C, C, kH, kW) tensor.

    Returns:
        (C, C, kH, kW) kernel tensor.
    """
    return torch.load(path, weights_only=True)


def fit_whitening_kernels(
    images: torch.Tensor,
    patch_size: int = 9,
    n_patches: int = 1_000_000,
    epsilon: float = 1e-2,
    rho: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit whitening kernels from image patches (Falez 2020 Eqs 8-12).

    Approximates ZCA whitening via cross-channel impulse response kernels.

    Args:
        images: (N, C, H, W) tensor of images in [0, 1].
        patch_size: Side length of square patches (must be odd).
        n_patches: Number of random patches to sample.
        epsilon: Regularization constant for eigenvalue inversion.
        rho: Fraction of eigenvalues to retain (0.15 per Falez 2020).

    Returns:
        (kernels, mean) where kernels is (C, C, kH, kW) for cross-channel conv
        and mean is (C * kH * kW,) patch mean vector.
    """
    N, C, H, W = images.shape
    dim = C * patch_size * patch_size

    # Cap n_patches to what's available
    max_patches = N * (H - patch_size + 1) * (W - patch_size + 1)
    n_patches = min(n_patches, max_patches)

    patches = _extract_patches(images, patch_size, n_patches)

    # Center patches (Eq 8)
    mean = patches.mean(dim=0)
    centered = patches - mean

    # Covariance matrix (Eq 9)
    cov = (centered.T @ centered) / n_patches

    # Eigen-decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Sort descending
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Retain top rho fraction of eigenvalues
    n_keep = max(1, int(rho * len(eigenvalues)))
    eigenvalues = eigenvalues[:n_keep]
    eigenvectors = eigenvectors[:, :n_keep]

    # ZCA whitening matrix: W = U @ diag(1/sqrt(Λ + ε)) @ U^T (Eq 9)
    scale = torch.diag(1.0 / torch.sqrt(eigenvalues + epsilon))
    zca_matrix = eigenvectors @ scale @ eigenvectors.T  # (dim, dim)

    # Extract cross-channel impulse response kernels (Eqs 11-12)
    # For each channel c, the kernel is the ZCA matrix's response to a
    # unit impulse at the center pixel of channel c.
    center_pixel = patch_size * patch_size // 2
    kernels = []
    for c in range(C):
        impulse_idx = c * patch_size * patch_size + center_pixel
        kernel_flat = zca_matrix[:, impulse_idx]  # (dim,) full cross-channel response
        kernel_3d = kernel_flat.reshape(C, patch_size, patch_size)  # (C, kH, kW)
        # Subtract filter mean (DC removal, per reference implementation)
        kernel_3d = kernel_3d - kernel_3d.mean()
        kernels.append(kernel_3d)

    kernels = torch.stack(kernels)  # (C, C, kH, kW)
    return kernels, mean


def apply_whitening_kernels(
    images: torch.Tensor,
    kernels: torch.Tensor,
    mean: torch.Tensor,
) -> torch.Tensor:
    """Apply pre-fitted whitening kernels to images via cross-channel convolution.

    The kernels already have DC removal built in (filter mean subtracted during
    fitting), so no additional bias subtraction is needed.

    Args:
        images: (N, C, H, W) tensor.
        kernels: (C, C, kH, kW) cross-channel convolution kernels.
        mean: (C * kH * kW,) patch mean vector from fitting (unused during
            application, kept for API compatibility).

    Returns:
        (N, C, H, W) whitened images (same spatial dimensions via padding).
    """
    kH = kernels.shape[2]
    padding = kH // 2
    # Replicate padding to match C++ clamp-to-edge boundary handling
    padded = F.pad(images, [padding, padding, padding, padding], mode="replicate")
    return F.conv2d(padded, kernels)
