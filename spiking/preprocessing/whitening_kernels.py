import torch
import torch.nn.functional as F


def _extract_patches(
    images: torch.Tensor, patch_size: int, n_patches: int
) -> torch.Tensor:
    """Extract random patches from images.

    Args:
        images: (N, C, H, W) tensor.
        patch_size: Side length of square patches.
        n_patches: Number of patches to extract.

    Returns:
        (n_patches, C * patch_size * patch_size) flattened patches.
    """
    N, C, H, W = images.shape
    half = patch_size // 2

    # Random image indices and spatial positions
    img_idx = torch.randint(0, N, (n_patches,))
    row_idx = torch.randint(half, H - half, (n_patches,))
    col_idx = torch.randint(half, W - half, (n_patches,))

    patches = []
    for i in range(n_patches):
        patch = images[
            img_idx[i],
            :,
            row_idx[i] - half : row_idx[i] + half + 1,
            col_idx[i] - half : col_idx[i] + half + 1,
        ]
        patches.append(patch.flatten())

    return torch.stack(patches)


def fit_whitening_kernels(
    images: torch.Tensor,
    patch_size: int = 9,
    n_patches: int = 1_000_000,
    epsilon: float = 1e-2,
    rho: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit whitening kernels from image patches (Falez 2020 Eqs 8-12).

    Approximates ZCA whitening via per-channel impulse response kernels.

    Args:
        images: (N, C, H, W) tensor of images in [0, 1].
        patch_size: Side length of square patches (must be odd).
        n_patches: Number of random patches to sample.
        epsilon: Regularization constant for eigenvalue inversion.
        rho: Fraction of eigenvalues to retain (1.0 = keep all).

    Returns:
        (kernels, mean) where kernels is (C, 1, kH, kW) for depthwise conv
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
    cov = (centered.T @ centered) / (n_patches - 1)

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

    # Extract per-channel impulse response kernels (Eqs 11-12)
    # For each channel c, the kernel is the ZCA matrix's response to a
    # unit impulse at the center pixel of channel c.
    center_pixel = patch_size * patch_size // 2
    kernels = []
    for c in range(C):
        impulse_idx = c * patch_size * patch_size + center_pixel
        kernel_flat = zca_matrix[:, impulse_idx]  # (dim,)
        # Extract only channel c's portion of the response
        c_start = c * patch_size * patch_size
        c_end = c_start + patch_size * patch_size
        kernel_2d = kernel_flat[c_start:c_end].reshape(1, patch_size, patch_size)
        kernels.append(kernel_2d)

    kernels = torch.stack(kernels)  # (C, 1, kH, kW)
    return kernels, mean


def apply_whitening_kernels(
    images: torch.Tensor,
    kernels: torch.Tensor,
    mean: torch.Tensor,
) -> torch.Tensor:
    """Apply pre-fitted whitening kernels to images via depthwise convolution.

    The mean vector captures the average patch content. Centering is done by
    convolving the mean's spatial structure with the whitening kernels to get
    a per-channel bias, then subtracting it from the convolution output.

    Args:
        images: (N, C, H, W) tensor.
        kernels: (C, 1, kH, kW) depthwise convolution kernels.
        mean: (C * kH * kW,) patch mean vector from fitting.

    Returns:
        (N, C, H, W) whitened images (same spatial dimensions via padding).
    """
    C = kernels.shape[0]
    kH = kernels.shape[2]
    padding = kH // 2

    # Convolve images with whitening kernels (depthwise)
    conv_out = F.conv2d(images, kernels, padding=padding, groups=C)

    # Compute bias from the mean: for each channel c, dot the mean's
    # channel-c portion with the kernel for channel c.
    # mean reshaped: (C, kH, kW)
    mean_spatial = mean.reshape(C, kH, kH)
    bias = (mean_spatial * kernels.squeeze(1)).sum(dim=(1, 2))  # (C,)

    whitened = conv_out - bias.reshape(1, C, 1, 1)
    return whitened
