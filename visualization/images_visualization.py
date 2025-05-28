import numpy as np
from matplotlib import pyplot as plt

from .consts import FIG_SIZE


class ImagesVisualization:
    @classmethod
    def plot_images(
        cls,
        *images: list[np.ndarray],
        nrows: int = 1,
        ncols: int = 1,
        title: str | None = None,
        titles: list[str] | None = None,
        cmap: str | None = None,
    ):
        if nrows * ncols < len(images):
            raise ValueError("Invalid nrows and ncols")
        if titles is not None and len(titles) != len(images):
            raise ValueError(
                "Mismatch between the number of images and the number of titles"
            )
        plt.figure(figsize=FIG_SIZE)
        plt.axis("off")
        if title:
            plt.title(title)
        for idx, image in enumerate(images):
            plt.subplot(nrows, ncols, idx + 1)
            plt.imshow(image, cmap=cmap)
            if titles:
                plt.title(titles[idx])
            plt.axis("off")
        plt.show()
