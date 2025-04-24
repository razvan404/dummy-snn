import torch
import torch.nn.functional as F


def gaussian_blur(img: torch.Tensor, sigma: float):
    # Create 1D Gaussian kernel
    size = int(8 * sigma + 1)
    if size % 2 == 0:
        size += 1
    coords = torch.arange(size, dtype=torch.float32, device=img.device) - size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)

    # Apply separable convolution: first horizontal, then vertical
    img = F.conv2d(img.unsqueeze(0), kernel.unsqueeze(2), padding=(0, size // 2))
    img = F.conv2d(img, kernel.unsqueeze(3), padding=(size // 2, 0))
    return img.squeeze(0)


def apply_difference_of_gaussians_filter(
    image: torch.Tensor, sigma_center: float = 1.0, sigma_surround: float = 2.0
) -> torch.Tensor:
    """
    Apply the Difference-of-Gaussians (DoG) filter to an image.
    Args:
        image: 2D tensor representing the grayscale image (H x W) or 3D tensor (C x H x W).
        sigma_center: Standard deviation for the center Gaussian.
        sigma_surround: Standard deviation for the surround Gaussian.
    Returns:
        Tensor of shape (2, H, W) containing DoG on/off channels.
    """
    if image.ndim == 2:
        image = image.unsqueeze(0)

    gaussian_center = gaussian_blur(image, sigma_center)
    gaussian_surround = gaussian_blur(image, sigma_surround)

    dog = gaussian_center - gaussian_surround
    x_on = torch.clamp(dog, min=0)
    x_off = torch.clamp(-dog, min=0)

    return torch.cat((x_on, x_off), dim=0)
