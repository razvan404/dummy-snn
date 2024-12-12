import numpy as np
from scipy.ndimage import gaussian_filter


def apply_difference_of_gaussians_filter(
    image: np.ndarray, sigma_center: float = 1.0, sigma_surround: float = 2.0
):
    """
    Apply the Difference-of-Gaussians (DoG) filter to an image.
    Args:
        image: 2D array representing the grayscale image.
        sigma_center: Variance for the center Gaussian kernel.
        sigma_surround: Variance for the surround Gaussian kernel.
    Returns:
        x_on: Positive DoG values (on channel).
        x_off: Negative DoG values (off channel).
    """
    gaussian_center = gaussian_filter(image, sigma=sigma_center)
    gaussian_surround = gaussian_filter(image, sigma=sigma_surround)

    dog = gaussian_center - gaussian_surround

    x_on = np.maximum(0, dog)
    x_off = np.maximum(0, -dog)

    return (
        np.stack((x_on, x_off), axis=0)
        if len(image.shape) == 2
        else np.concatenate((x_on, x_off), axis=0)
    )
